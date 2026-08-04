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
    return torch.full(
        (probs.shape[0],),
        value,
        dtype=dtype,
        device=probs.device,
    )


def _normalize_filtered_probs(filtered_probs: torch.Tensor) -> torch.Tensor:
    normalizer = filtered_probs.sum(dim=-1, keepdim=True)
    return torch.where(
        normalizer > 0,
        filtered_probs / normalizer,
        torch.zeros_like(filtered_probs),
    )


def top_k_renorm_probs_torch(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    """Rank-based top-k filtering followed by per-row renormalization."""
    assert probs.ndim == 2
    probs = probs.float()
    batch_size, vocab_size = probs.shape
    if batch_size == 0:
        return probs.clone()
    assert vocab_size > 0

    top_ks = _per_row_threshold(top_k, probs=probs, dtype=torch.int64)
    top_ks = top_ks.clamp(min=1, max=vocab_size)

    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    ranks = torch.arange(vocab_size, device=probs.device).view(1, -1)
    sorted_probs = sorted_probs.masked_fill(ranks >= top_ks.view(-1, 1), 0.0)
    sorted_probs = _normalize_filtered_probs(sorted_probs)
    return torch.zeros_like(sorted_probs).scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_probs,
    )


def top_p_renorm_probs_torch(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Nucleus filtering followed by per-row renormalization."""
    assert probs.ndim == 2
    probs = probs.float()
    batch_size, vocab_size = probs.shape
    if batch_size == 0:
        return probs.clone()
    assert vocab_size > 0

    top_ps = _per_row_threshold(top_p, probs=probs, dtype=torch.float32)
    top_ps = top_ps.clamp(min=0.0, max=1.0)

    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    prefix_mass = sorted_probs.cumsum(dim=-1) - sorted_probs
    sorted_probs = sorted_probs.masked_fill(
        prefix_mass > top_ps.view(-1, 1),
        0.0,
    )
    sorted_probs = _normalize_filtered_probs(sorted_probs)
    return torch.zeros_like(sorted_probs).scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_probs,
    )
