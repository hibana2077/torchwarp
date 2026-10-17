"""Fast Free Viewpoint Matching (FVM), Eq. (13) of the JEANIE paper.

A viewpoint SoftMin at every temporal pair, then soft-DTW on the result.
"""

import torch

from .. import sdtw  # module, not the function: sdtw imports this package
from .distances import euclidean_cost, rbf_cost, squared_euclidean_cost

_METRICS = {"euclidean": euclidean_cost, "sqeuclidean": squared_euclidean_cost,
            "rbf": rbf_cost}


def fvm_from_cost(cost, *, gamma=0.1, backend="auto"):
    """FVM on a precomputed cost tensor.

    Args:
        cost: [B, K..., T, U]; every dim between the batch dim and [T, U] is a
            viewpoint index reduced by SoftMin, e.g. [B, K, T, U] (query
            viewpoints), [B, K_query, K_support, T, U] or
            [B, K1, K2, T, U].
        gamma: SoftMin temperature (> 0).
        backend: "auto" (CUDA kernel when possible), "cuda" or "torch".

    Returns:
        distance [B]
    """
    if not cost.is_floating_point():
        raise TypeError("cost must be a floating-point tensor")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if cost.ndim < 4:
        raise ValueError("cost must be [B, K..., T, U] with at least one viewpoint dim")
    B, T, U = cost.shape[0], cost.shape[-2], cost.shape[-1]
    flat = cost.reshape(B, -1, T, U)
    temporal = -gamma * torch.logsumexp(-flat / gamma, dim=1)
    return sdtw.soft_dtw_from_cost(temporal, gamma=gamma, backend=backend)


def fvm(query, support, *, gamma=0.1, metric="euclidean", backend="auto"):
    """FVM between viewpoint-augmented queries and supports.

    Args:
        query: [B, K..., T, D] with one or more viewpoint dims.
        support: [B, U, D], or [B, K'..., U, D] with its own viewpoint dims.
        gamma: SoftMin temperature (> 0).
        metric: Base distance: "euclidean" (default), "sqeuclidean" or "rbf".
        backend: "auto" (CUDA kernel when possible), "cuda" or "torch".

    Returns:
        distance [B]
    """
    if metric not in _METRICS:
        raise ValueError("metric must be one of {}".format(tuple(_METRICS)))
    if query.ndim < 4 or support.ndim < 3:
        raise ValueError("query must be [B, K..., T, D] and support [B, U, D] or "
                         "[B, K'..., U, D]")
    B, T, D = query.shape[0], query.shape[-2], query.shape[-1]
    U = support.shape[-2]
    q = query.reshape(B, -1, T, D)
    if support.ndim == 3:
        cost = _METRICS[metric](q, support)                      # [B, K, T, U]
    else:
        s = support.reshape(B, -1, U, D)
        cost = _METRICS[metric](q.unsqueeze(2), s.unsqueeze(1))  # [B, K, K', T, U]
    return fvm_from_cost(cost, gamma=gamma, backend=backend)
