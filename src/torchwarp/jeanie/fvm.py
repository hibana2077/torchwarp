"""Fast Free Viewpoint Matching (FVM), Eq. (13) of the JEANIE paper.

A viewpoint SoftMin at every temporal pair, then soft-DTW on the result.
Each wrapper accepts its unbatched layout or one extra leading batch dim.
"""

import torch

from .alignment import soft_dtw


def fvm_from_cost(
    cost: torch.Tensor,
    gamma: float = 0.1,
    batched: bool = False,
    backend: str = "auto",
) -> torch.Tensor:
    """General FVM: every dim except the last two (and the batch dim when
    ``batched``) is a viewpoint index reduced by SoftMin.

    Unbatched layouts: [K,T,U], [K1,K2,T,U] / [Kq,Ks,T,U], [Kq1,Kq2,Ks1,Ks2,T,U].
    """
    if not cost.is_floating_point():
        raise TypeError("cost must be a floating-point tensor")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    lead = 1 if batched else 0
    if cost.ndim < lead + 3:
        raise ValueError("cost needs at least one viewpoint dim plus [T, U]")
    T, U = cost.shape[-2:]
    flat = cost.reshape(*cost.shape[:lead], -1, T, U)
    temporal = -gamma * torch.logsumexp(-flat / gamma, dim=lead)
    return soft_dtw(temporal, gamma, backend)


def _fvm_fixed(cost, ndim, layout, gamma, backend):
    if cost.ndim not in (ndim, ndim + 1):
        raise ValueError("cost must have shape {} (optionally with a batch dim)".format(layout))
    return fvm_from_cost(cost, gamma, batched=cost.ndim == ndim + 1, backend=backend)


def fvm_query_only_1d(cost, gamma=0.1, backend="auto"):
    """Query-only 1-D FVM for cost [K, T, U] or [B, K, T, U]."""
    return _fvm_fixed(cost, 3, "[K, T, U]", gamma, backend)


def fvm_query_only_2d(cost, gamma=0.1, backend="auto"):
    """Query-only 2-D FVM for cost [K1, K2, T, U] or [B, K1, K2, T, U]."""
    return _fvm_fixed(cost, 4, "[K1, K2, T, U]", gamma, backend)


def fvm_1d_from_cost(cost, gamma=0.1, backend="auto"):
    """Full 1-D FVM for cost [Kq, Ks, T, U] or [B, Kq, Ks, T, U]."""
    return _fvm_fixed(cost, 4, "[K_query, K_support, T, U]", gamma, backend)


def fvm_2d_from_cost(cost, gamma=0.1, backend="auto"):
    """Full 2-D FVM for cost [Kq1,Kq2,Ks1,Ks2,T,U] or with a batch dim."""
    return _fvm_fixed(cost, 6, "[Kq1,Kq2,Ks1,Ks2,T,U]", gamma, backend)
