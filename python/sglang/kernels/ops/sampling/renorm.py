"""Portable top-k / top-p probability renormalization.

Matches the threshold semantics of the FlashInfer-derived AOT kernels these stand
in for: locate the pivot entry that the requested budget reaches, keep every entry
``>= pivot`` -- so ties at the pivot are all retained -- and renormalize. A
rank-based cutoff would instead keep exactly k entries and break ties by sort
order, which diverges from CUDA whenever probabilities tie at the boundary.
"""

from typing import Union

import torch


def _per_row_threshold(
    value: Union[torch.Tensor, int, float],
    *,
    probs: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        value = value.to(device=probs.device, dtype=dtype).reshape(-1)
        assert value.numel() in (1, probs.shape[0])
        if value.numel() == 1:
            value = value.expand(probs.shape[0])
        return value
    return torch.full((probs.shape[0],), value, dtype=dtype, device=probs.device)


def _apply_pivot(probs: torch.Tensor, pivots: torch.Tensor) -> torch.Tensor:
    kept = torch.where(probs >= pivots.unsqueeze(1), probs, torch.zeros_like(probs))
    normalizer = kept.sum(dim=-1, keepdim=True)
    return torch.where(normalizer > 0, kept / normalizer, torch.zeros_like(kept))


def _top_k_pivots(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    """Value of the k-th largest entry in each row."""
    descending, _ = torch.sort(probs, dim=-1, descending=True)
    return descending.gather(1, (top_ks - 1).unsqueeze(1)).squeeze(1)


def _top_p_pivots(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Pivot of the nucleus: the least likely entry that is still kept.

    Accumulates the discarded tail upwards from the smallest entry rather than
    testing ``cumsum >= top_p`` from the top. A row of float32 probabilities sums to
    slightly under one, so a descending scan lets the leading terms round up to one
    on their own and truncate the tail of a peaked row at ``top_p=1``.
    """
    ascending, _ = torch.sort(probs, dim=-1)
    cutoff = torch.searchsorted(
        ascending.cumsum(dim=-1).contiguous(),
        (1.0 - top_ps).unsqueeze(1).contiguous(),
        right=False,
    ).squeeze(1)
    cutoff = cutoff.clamp(max=probs.shape[1] - 1)
    return ascending.gather(1, cutoff.unsqueeze(1)).squeeze(1)


def top_k_renorm_probs_torch(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    """Keep every entry at least as likely as the k-th largest, then renormalize."""
    assert probs.ndim == 2
    probs = probs.float()
    batch_size, vocab_size = probs.shape
    if batch_size == 0:
        return probs.clone()
    assert vocab_size > 0

    top_ks = _per_row_threshold(probs=probs, value=top_k, dtype=torch.int64).clamp(
        1, vocab_size
    )
    return _apply_pivot(probs, _top_k_pivots(probs, top_ks))


def top_p_renorm_probs_torch(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Keep the nucleus -- every entry at least as likely as its pivot -- and
    renormalize."""
    assert probs.ndim == 2
    probs = probs.float()
    batch_size, vocab_size = probs.shape
    if batch_size == 0:
        return probs.clone()
    assert vocab_size > 0

    top_ps = _per_row_threshold(probs=probs, value=top_p, dtype=torch.float32).clamp(
        0.0, 1.0
    )
    return _apply_pivot(probs, _top_p_pivots(probs, top_ps))
