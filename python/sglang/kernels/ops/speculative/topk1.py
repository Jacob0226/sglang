from __future__ import annotations

import torch
import triton
import triton.language as tl

_DRAFT_TOPK1_BLOCK = 8192


@triton.jit
def _draft_topk1_partial_argmax_kernel(
    logits,
    partial_vals,
    partial_indices,
    logits_row_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # int64 row base: row * stride overflows int32 once bs * vocab reaches 2^31.
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    vals = tl.load(
        logits + row * logits_row_stride + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    # Keep NaNs on valid lanes from selecting the masked tail.
    vals = tl.where(vals == vals, vals, -1e30)

    max_val = tl.max(vals, axis=0)
    local_index = tl.argmax(vals, axis=0)
    out_offset = row * num_splits + split
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + local_index)


@triton.jit
def _draft_topk1_finalize_kernel(
    partial_vals,
    partial_indices,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    num_splits: tl.constexpr,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )

    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)
    if WRITE_DRAFT_TOKEN:
        tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)

    position = tl.load(positions + row)
    tl.store(positions + row, position + 1)


def draft_topk1_postprocess(
    next_token_logits: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
):
    """Argmax draft logits for topk=1 and advance positions.

    PyTorch eager argmax reduces each row with too little parallelism for the
    GLM/DSV4 vocab widths in CUDA graph replay. This split reduction exposes
    the vocab dimension across CTAs, then finalizes one token per row.

    If ``draft_tokens`` is given, the finalize kernel also stores the argmax
    into ``draft_tokens[:, draft_token_column]``, mutating the caller-owned
    buffer in place. ``topk_p`` is returned as constant 1.0: topk=1 drafting
    is greedy and the chain probabilities are unused downstream.
    """
    assert next_token_logits.ndim == 2
    assert next_token_logits.stride(1) == 1
    assert positions.ndim == 1
    assert positions.is_contiguous()
    assert positions.shape[0] == next_token_logits.shape[0]
    assert positions.device == next_token_logits.device
    write_draft_token = draft_tokens is not None
    if write_draft_token:
        assert draft_tokens.ndim == 2
        assert draft_tokens.dtype == torch.long
        assert draft_tokens.device == next_token_logits.device
        assert draft_tokens.shape[0] == next_token_logits.shape[0]
        assert draft_tokens.stride(1) == 1
        assert 0 <= draft_token_column < draft_tokens.shape[1]

    bs, vocab_size = next_token_logits.shape
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=next_token_logits.device)
    topk_index = torch.empty(
        (bs, 1), dtype=torch.int64, device=next_token_logits.device
    )
    if bs == 0:
        return topk_p, topk_index

    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty(
        (bs, num_splits), dtype=torch.float32, device=next_token_logits.device
    )
    partial_indices = torch.empty(
        (bs, num_splits), dtype=torch.int32, device=next_token_logits.device
    )

    _draft_topk1_partial_argmax_kernel[(bs, num_splits)](
        next_token_logits,
        partial_vals,
        partial_indices,
        next_token_logits.stride(0),
        vocab_size,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    # Dummy operand for the disabled draft-token slot: the pointer must be
    # valid even though the kernel never dereferences it (gated off by
    # WRITE_DRAFT_TOKEN).
    _draft_topk1_finalize_kernel[(bs,)](
        partial_vals,
        partial_indices,
        topk_p,
        topk_index,
        positions,
        draft_tokens if write_draft_token else topk_index,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_token_column,
        num_splits,
        WRITE_DRAFT_TOKEN=write_draft_token,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return topk_p, topk_index


_SHARD_CANDIDATE_WIDTH = 4


@triton.jit
def _draft_topk1_shard_candidate_kernel(
    partial_vals,
    partial_indices,
    candidates,
    col_offset,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )
    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split) + col_offset
    tl.store(candidates + row * WIDTH, tl.max(vals, axis=0))
    tl.store(candidates + row * WIDTH + 1, index.to(tl.float32))


@triton.jit
def _draft_topk1_rank_finalize_kernel(
    candidates,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    num_rows,
    world_size: tl.constexpr,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    ranks = tl.arange(0, BLOCK)
    mask = ranks < world_size
    vals = tl.load(
        candidates + (ranks * num_rows + row) * WIDTH,
        mask=mask,
        other=-float("inf"),
    )
    # Ranks hold ascending vocab ranges, so the lowest rank on a tie is the
    # lowest token id, as in draft_topk1_postprocess.
    rank = tl.argmax(vals, axis=0)
    index = tl.load(candidates + (rank * num_rows + row) * WIDTH + 1).to(tl.int64)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)
    if WRITE_DRAFT_TOKEN:
        tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)

    position = tl.load(positions + row)
    tl.store(positions + row, position + 1)


def draft_topk1_postprocess_sharded(
    shard_logits: torch.Tensor,
    vocab_shard,
    positions: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
):
    """``draft_topk1_postprocess`` over this TP rank's vocab shard.

    Each rank reduces its shard to one ``(max, token id)`` per row and only
    those are all-gathered, instead of the full logits. Picks the same token as
    the gathered path, ties included.
    """
    from sglang.srt.runtime_context import get_parallel

    assert shard_logits.ndim == 2
    assert shard_logits.stride(1) == 1
    assert positions.ndim == 1 and positions.is_contiguous()
    assert positions.shape[0] == shard_logits.shape[0]
    assert vocab_shard.vocab_size < 2**24
    write_draft_token = draft_tokens is not None
    if write_draft_token:
        assert draft_tokens.dtype == torch.long
        assert draft_tokens.shape[0] == shard_logits.shape[0]
        assert draft_tokens.stride(1) == 1

    device = shard_logits.device
    bs = shard_logits.shape[0]
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=device)
    topk_index = torch.empty((bs, 1), dtype=torch.int64, device=device)
    if bs == 0:
        return topk_p, topk_index

    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(vocab_shard.valid, block)
    partial_vals = torch.empty((bs, num_splits), dtype=torch.float32, device=device)
    partial_indices = torch.empty((bs, num_splits), dtype=torch.int32, device=device)
    _draft_topk1_partial_argmax_kernel[(bs, num_splits)](
        shard_logits,
        partial_vals,
        partial_indices,
        shard_logits.stride(0),
        vocab_shard.valid,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    # Rows padded to 16 bytes for the custom all-gather.
    candidates = torch.zeros(
        (bs, _SHARD_CANDIDATE_WIDTH), dtype=torch.float32, device=device
    )
    _draft_topk1_shard_candidate_kernel[(bs,)](
        partial_vals,
        partial_indices,
        candidates,
        vocab_shard.start,
        num_splits,
        BLOCK=triton.next_power_of_2(num_splits),
        WIDTH=_SHARD_CANDIDATE_WIDTH,
        num_warps=1,
    )
    group = get_parallel().tp_group
    world_size = group.world_size
    gathered = torch.empty(
        (world_size * bs, _SHARD_CANDIDATE_WIDTH), dtype=torch.float32, device=device
    )
    group.all_gather_into_tensor(gathered, candidates)
    _draft_topk1_rank_finalize_kernel[(bs,)](
        gathered,
        topk_p,
        topk_index,
        positions,
        draft_tokens if write_draft_token else topk_index,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_token_column,
        bs,
        world_size,
        WRITE_DRAFT_TOKEN=write_draft_token,
        BLOCK=triton.next_power_of_2(world_size),
        WIDTH=_SHARD_CANDIDATE_WIDTH,
        num_warps=1,
    )
    return topk_p, topk_index


def draft_topk1_argmax_only(next_token_logits: torch.Tensor):
    """Select top-k=1 without mutating caller-owned positions."""
    # The upstream finalizer always advances positions. Draft-extend has already
    # consumed its positions, so absorb that side effect in a discarded buffer.
    scratch_positions = torch.zeros(
        next_token_logits.shape[0],
        dtype=torch.long,
        device=next_token_logits.device,
    )
    return draft_topk1_postprocess(next_token_logits, scratch_positions)
