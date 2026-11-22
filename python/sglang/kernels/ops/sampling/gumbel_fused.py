"""One-sample exponential race (Gumbel-max) over ``probs`` rows in two launches.

Replaces exponential_ / clamp / divide / argmax / gather (and the greedy-row argmax /
where / gather) of the unseeded draft proposal path: the PyTorch argmax reduces
a row with too little parallelism at vocabulary widths, and each of the other ops
is its own kernel in graph replay.
"""

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

_BLOCK = 8192
_TINY = 1.1754943508222875e-38

# Per-device philox key, advanced by the finalize kernel after every draw. It
# starts from the generator's initial seed, so every TP rank draws the same
# variates, and it lives outside any graph pool so replays keep advancing it.
_SEED_STATE: Dict[int, torch.Tensor] = {}


def _seed_state(device: torch.device) -> Optional[torch.Tensor]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    state = _SEED_STATE.get(index)
    if state is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        initial = torch.cuda.default_generators[index].initial_seed() & 0x7FFFFFFF
        state = torch.full((1,), initial, dtype=torch.int32, device=device)
        _SEED_STATE[index] = state
    return state


@triton.jit
def _race_partial_kernel(
    probs,
    probs_row_stride,
    seed_ptr,
    top_ks,
    partial_vals,
    partial_indices,
    vocab_size,
    num_splits: tl.constexpr,
    HAS_TOP_KS: tl.constexpr,
    BLOCK: tl.constexpr,
    TINY: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    p = tl.load(probs + row * probs_row_stride + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    greedy = False
    if HAS_TOP_KS:
        greedy = tl.load(top_ks + row) <= 1
    if greedy:
        scores = p
    else:
        seed = tl.load(seed_ptr)
        u = tl.rand(seed, (row * vocab_size + offsets).to(tl.int32))
        q = tl.maximum(-tl.log(tl.maximum(u, TINY)), TINY)
        scores = p / q
    # Keep NaNs and the masked tail from winning.
    scores = tl.where(mask & (scores == scores), scores, -1.0)
    best = tl.max(scores, axis=0)
    index = tl.argmax(scores, axis=0)
    out = row * num_splits + split
    tl.store(partial_vals + out, best)
    tl.store(partial_indices + out, split * BLOCK + index)


@triton.jit
def _race_finalize_kernel(
    probs,
    probs_row_stride,
    partial_vals,
    partial_indices,
    sample_p,
    sample_index,
    seed_ptr,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    num_splits: tl.constexpr,
    ADVANCE_SEED: tl.constexpr,
    ADVANCE_POSITIONS: tl.constexpr,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=offsets < num_splits,
        other=-float("inf"),
    )
    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(sample_index + row, index)
    tl.store(sample_p + row, tl.load(probs + row * probs_row_stride + index))
    if WRITE_DRAFT_TOKEN:
        tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)
    if ADVANCE_POSITIONS:
        tl.store(positions + row, tl.load(positions + row) + 1)
    if ADVANCE_SEED:
        if row == 0:
            tl.store(seed_ptr, (tl.load(seed_ptr) + 1) & 0x7FFFFFFF)


def race_sample(
    probs: torch.Tensor,
    top_ks: Optional[torch.Tensor] = None,
    *,
    positions: Optional[torch.Tensor] = None,
    draft_tokens: Optional[torch.Tensor] = None,
    draft_token_column: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draw X ~ probs per row (rows with ``top_ks <= 1`` take the argmax instead).

    Returns ``(probs[X], X)``, each ``[rows, 1]`` (fp32 / int64). The exponential
    variates are keyed by a per-device counter that starts at the default CUDA
    generator's initial seed and advances on device after each draw, so graph
    replays draw fresh values and every TP rank draws the same ones.

    ``positions`` (if given) is advanced by one in place, and X is also stored
    into ``draft_tokens[:, draft_token_column]``, saving the separate launches.
    """
    assert probs.ndim == 2 and probs.stride(1) == 1
    rows, vocab_size = probs.shape
    sample_p = torch.empty((rows, 1), dtype=torch.float32, device=probs.device)
    sample_index = torch.empty((rows, 1), dtype=torch.int64, device=probs.device)
    if rows == 0:
        return sample_p, sample_index
    if positions is not None:
        assert positions.is_contiguous() and positions.shape[0] == rows
    if draft_tokens is not None:
        assert draft_tokens.dtype == torch.int64 and draft_tokens.stride(1) == 1
        assert draft_tokens.shape[0] == rows
    num_splits = triton.cdiv(vocab_size, _BLOCK)
    seed = _seed_state(probs.device)
    advance_seed = seed is not None
    if seed is None:
        seed = torch.randint(
            0, 2**31 - 1, (1,), dtype=torch.int32, device=probs.device
        )
    partial_vals = torch.empty(
        (rows, num_splits), dtype=torch.float32, device=probs.device
    )
    partial_indices = torch.empty(
        (rows, num_splits), dtype=torch.int32, device=probs.device
    )
    _race_partial_kernel[(rows, num_splits)](
        probs,
        probs.stride(0),
        seed,
        top_ks if top_ks is not None else seed,
        partial_vals,
        partial_indices,
        vocab_size,
        num_splits,
        HAS_TOP_KS=top_ks is not None,
        BLOCK=_BLOCK,
        TINY=_TINY,
        num_warps=8,
    )
    _race_finalize_kernel[(rows,)](
        probs,
        probs.stride(0),
        partial_vals,
        partial_indices,
        sample_p,
        sample_index,
        seed,
        positions if positions is not None else sample_index,
        draft_tokens if draft_tokens is not None else sample_index,
        draft_tokens.stride(0) if draft_tokens is not None else 0,
        draft_token_column,
        num_splits,
        ADVANCE_SEED=advance_seed,
        ADVANCE_POSITIONS=positions is not None,
        WRITE_DRAFT_TOKEN=draft_tokens is not None,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return sample_p, sample_index
