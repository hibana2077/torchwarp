"""Alignment paths for visualisation.

Every distance in torchwarp is a soft-min over warping paths. Its gradient
with respect to the cost tensor is the expected path occupancy: entry
(t, u) (or (k, t, u) with a viewpoint axis) is the probability that the
soft path visits that cell. These functions return that occupancy
(``soft``), a hard path decoded from it (``path``) and the distance.

uDTW additionally returns the pairwise variance Sigma, and JEANIE / FVM a
viewpoint per path step, so the figures of the uDTW and JEANIE papers can
be reproduced with ``torchwarp.plot``.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from .jeanie import (euclidean_cost, fvm_query_only_1d, jeanie_1d_from_cost, soft_dtw,
                     squared_euclidean_cost)
from .udtw import pairwise_matrices, udtw_from_matrices

__all__ = ["Alignment", "sdtw", "sdtw_per_view", "udtw", "fvm", "jeanie", "decode"]


@dataclass
class Alignment:
    """Result of a path computation (batched over the first dimension).

    Attributes:
        distance: [B] distance value(s).
        soft: [B, T, U] or [B, K, T, U] expected path occupancy.
        path: per batch item, a list of (t, u) or (t, u, k) cells from the
            start (0, 0) to the end (T-1, U-1).
        cost: the cost tensor the paths were computed on.
        variance: [B, T, U] pairwise variance Sigma (uDTW only).
        penalty: [B] uncertainty penalty Omega (uDTW only).
        angles: viewpoint labels for the K axis, if given.
    """

    distance: torch.Tensor
    soft: torch.Tensor
    path: List[List[Tuple[int, ...]]]
    cost: torch.Tensor
    variance: Optional[torch.Tensor] = None
    penalty: Optional[torch.Tensor] = None
    angles: Optional[list] = field(default=None)


def _occupancy(fn, cost):
    """distance = fn(cost) and its gradient d distance / d cost."""
    with torch.enable_grad():
        c = cost.detach().clone().requires_grad_(True)
        d = fn(c)
        (soft,) = torch.autograd.grad(d.sum(), c)
    return d.detach(), soft.detach()


def _decode_2d(occ):
    """Hard path through a [T, U] occupancy, from the end cell backwards."""
    T, U = occ.shape
    occ = occ.cpu()
    t, u, path = T - 1, U - 1, [(T - 1, U - 1)]
    while (t, u) != (0, 0):
        steps = [(t - 1, u - 1), (t - 1, u), (t, u - 1)]
        steps = [(a, b) for a, b in steps if a >= 0 and b >= 0]
        t, u = max(steps, key=lambda s: float(occ[s]))
        path.append((t, u))
    return path[::-1]


def _decode_3d(occ, max_shift):
    """Hard path through a [K, T, U] occupancy; the viewpoint may change by at
    most ``max_shift`` per step (``None``: any change)."""
    K, T, U = occ.shape
    occ = occ.cpu()
    k = int(occ[:, T - 1, U - 1].argmax())
    t, u, path = T - 1, U - 1, [(T - 1, U - 1, k)]
    shift = K if max_shift is None else max_shift
    while (t, u) != (0, 0):
        best, best_val = None, -float("inf")
        for a, b in [(t - 1, u - 1), (t - 1, u), (t, u - 1)]:
            if a < 0 or b < 0:
                continue
            for kk in range(max(0, k - shift), min(K, k + shift + 1)):
                v = float(occ[kk, a, b])
                if v > best_val:
                    best, best_val = (a, b, kk), v
        t, u, k = best
        path.append(best)
    return path[::-1]


def decode(soft, max_shift=None):
    """Hard paths for a batch of occupancies [B, T, U] or [B, K, T, U]."""
    if soft.ndim == 3:
        return [_decode_2d(s) for s in soft]
    return [_decode_3d(s, max_shift) for s in soft]


def _batched(x, ndim):
    return (x.unsqueeze(0), True) if x.ndim == ndim else (x, False)


def _base_cost(query, support, metric):
    if metric == "euclidean":
        return euclidean_cost(query, support)
    if metric == "sqeuclidean":
        return squared_euclidean_cost(query, support)
    raise ValueError("metric must be 'euclidean' or 'sqeuclidean'")


def sdtw(cost, gamma=0.1):
    """soft-DTW paths for a cost [B, T, U] (or [T, U])."""
    c, _ = _batched(cost, 2)
    d, soft = _occupancy(lambda x: soft_dtw(x, gamma), c)
    return Alignment(d, soft, decode(soft), c)


def sdtw_per_view(query, support, gamma=0.1, metric="euclidean", angles=None):
    """soft-DTW applied separately to every viewpoint of the query.

    query [B, K, T, D] (or [K, T, D]), support [B, U, D] (or [U, D]).
    Returns a list of K Alignments; path cells are (t, u, k).
    """
    q, _ = _batched(query, 3)
    s, _ = _batched(support, 2)
    cost = _base_cost(q, s, metric)
    out = []
    for k in range(cost.shape[1]):
        a = sdtw(cost[:, k], gamma)
        a.path = [[(t, u, k) for t, u in p] for p in a.path]
        a.angles = angles
        out.append(a)
    return out


def udtw(X, Y, sigma_x, sigma_y, gamma=0.1, beta=1.0):
    """uDTW paths. X [B, N, D], Y [B, M, D], sigma_x [B, N, 1], sigma_y [B, M, 1].

    ``soft`` is the path occupancy of the uncertainty-weighted distance, and
    ``variance`` the pairwise Sigma; ``penalty`` is the beta-weighted Omega.
    """
    with torch.no_grad():
        cost, pen, var = pairwise_matrices(X, Y, sigma_x, sigma_y, beta)

    def dist(c):
        return udtw_from_matrices(c, pen, gamma)[0]

    d, soft = _occupancy(dist, cost)
    with torch.no_grad():
        omega = udtw_from_matrices(cost, pen, gamma)[1]
    return Alignment(d, soft, decode(soft), cost, variance=var, penalty=omega)


def fvm(query, support, gamma=0.1, metric="euclidean", angles=None):
    """Query-only 1-D FVM paths; query [B, K, T, D], support [B, U, D]."""
    q, _ = _batched(query, 3)
    s, _ = _batched(support, 2)
    cost = _base_cost(q, s, metric)
    d, soft = _occupancy(lambda x: fvm_query_only_1d(x, gamma), cost)
    return Alignment(d, soft, decode(soft, max_shift=None), cost, angles=angles)


def jeanie(query, support, gamma=0.1, max_shift=1, metric="euclidean", angles=None,
           start_view=None):
    """JEANIE-1D paths; query [B, K, T, D], support [B, U, D].

    ``start_view`` restricts the path to start at that viewpoint index (the
    JEANIE paper shows one path per starting viewpoint).
    """
    q, _ = _batched(query, 3)
    s, _ = _batched(support, 2)
    cost = _base_cost(q, s, metric)
    if start_view is not None:
        block = torch.zeros_like(cost)
        block[:, :, 0, 0] = 1e4
        block[:, start_view, 0, 0] = 0
        cost = cost + block
    d, soft = _occupancy(lambda x: jeanie_1d_from_cost(x, gamma, max_shift), cost)
    return Alignment(d, soft, decode(soft, max_shift=max_shift), cost, angles=angles)
