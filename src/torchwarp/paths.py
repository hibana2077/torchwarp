"""Paths for visualisation, in the notation of the uDTW and JEANIE papers.

Each distance is a SoftMin_gamma over the path distances w_i = <Pi_i, D> of
all paths Pi_i of the transportation plan. For a pair of sequences these
functions return

* ``path_matrix``: SoftMinSel_gamma(w, [Pi_i]), the path matrices of all
  paths weighted by the soft-selector over their path distances (uDTW
  Eq. 3 with the path matrices in place of log Sigma). It is the plot of
  uDTW Fig. 2 (a)-(d) and tends to the path matrix Pi_{i*},
  i* = arg min_i w_i, as gamma -> 0. It equals the gradient of the distance
  with respect to D, which is how it is computed.
* ``optimal_path``: the path with the smallest distance, i* = arg min_i w_i,
  as a list of cells (JEANIE Fig. 7 draws it).

``torchwarp.plot`` draws both.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from . import sdtw as _sdtw
from .jeanie import euclidean_cost, fvm_query_only_1d, jeanie_1d_from_cost, squared_euclidean_cost
from .udtw import pairwise_matrices, udtw_from_cost

__all__ = ["Alignment", "soft_dtw", "soft_dtw_per_view", "udtw", "fvm", "jeanie", "optimal_path"]


@dataclass
class Alignment:
    """Result of a path computation (batched over the first dimension).

    Attributes:
        distance: [B] distance d.
        path_matrix: [B, T, U] (or [B, K, T, U] with K viewpoints)
            SoftMinSel_gamma(w, [Pi_i]).
        optimal_path: per batch item, the cells (t, u) (or (t, u, k)) of the
            path with the smallest distance, from (0, 0) to (T-1, U-1).
        cost: the cost matrix (tensor) the paths are evaluated on; for uDTW
            it is D * Sigma^dagger (uDTW Eq. 2).
        Sigma: [B, T, U] pair-wise variances (uDTW only).
        Omega: [B] penalty Omega(Sigma), times beta (uDTW only).
        angles: viewpoint labels for the K axis, if given.
    """

    distance: torch.Tensor
    path_matrix: torch.Tensor
    optimal_path: List[List[Tuple[int, ...]]]
    cost: torch.Tensor
    Sigma: Optional[torch.Tensor] = None
    Omega: Optional[torch.Tensor] = None
    angles: Optional[list] = field(default=None)

    def plot(self, type="alignment", **kwargs):
        """Plot this alignment with ``torchwarp.plot`` (needs matplotlib).

        type: "alignment" (path matrix), "uncertainty" (Sigma on the path,
        uDTW only), "twoway" (two 1-D sequences joined along the optimal
        path; pass ``x`` and ``y``) or "path_3d" (optimal path in 3D, for
        viewpoint alignments): the name of the ``torchwarp.plot`` function.
        Keyword arguments go to that function, e.g.
        ``index`` (batch item) and ``ax``. Returns the matplotlib axis.
        """
        from . import plot

        fns = {"alignment": plot.alignment, "uncertainty": plot.uncertainty,
               "twoway": plot.twoway, "path_3d": plot.path_3d}
        if type not in fns:
            raise ValueError("type must be one of {}".format(sorted(fns)))
        return fns[type](self, **kwargs)


def _value_and_grad(fn, D):
    """d = fn(D) and its gradient with respect to D."""
    with torch.enable_grad():
        c = D.detach().clone().requires_grad_(True)
        d = fn(c)
        (g,) = torch.autograd.grad(d.sum(), c)
    return d.detach(), g.detach()


def _optimal_3d(D, max_shift, start_view=None):
    """arg min over the paths of a [K, T, U] distance tensor (gamma = 0).

    The viewpoint may change by at most ``max_shift`` per step; the path
    starts at any viewpoint (or at ``start_view``) and ends at any viewpoint.
    """
    D = D.detach().double().cpu().numpy()
    K, T, U = D.shape
    r = np.full((K, T, U), np.inf)
    r[:, 0, 0] = D[:, 0, 0]
    if start_view is not None:
        r[:, 0, 0] = np.inf
        r[start_view, 0, 0] = D[start_view, 0, 0]
    back = np.zeros((K, T, U, 3), dtype=np.int64)
    for t in range(T):
        for u in range(U):
            if t == 0 and u == 0:
                continue
            for k in range(K):
                best, arg = np.inf, None
                for kk in range(max(0, k - max_shift), min(K, k + max_shift + 1)):
                    for a, b in ((t - 1, u - 1), (t - 1, u), (t, u - 1)):
                        if a >= 0 and b >= 0 and r[kk, a, b] < best:
                            best, arg = r[kk, a, b], (kk, a, b)
                if arg is not None:
                    r[k, t, u] = D[k, t, u] + best
                    back[k, t, u] = arg
    k, t, u = int(r[:, T - 1, U - 1].argmin()), T - 1, U - 1
    path = [(t, u, k)]
    while (t, u) != (0, 0):
        k, t, u = (int(v) for v in back[k, t, u])
        path.append((t, u, k))
    return path[::-1]


def optimal_path(cost, *, max_shift=0, start_view=None):
    """Paths with the smallest distance for a batch of cost matrices.

    cost: [B, T, U] (cells (t, u)) or [B, K, T, U] (cells (t, u, k)).
    """
    if cost.ndim == 3:
        return [[(t, u) for t, u, _ in _optimal_3d(c.unsqueeze(0), 0)] for c in cost]
    return [_optimal_3d(c, max_shift, start_view) for c in cost]


def _batched(x, ndim):
    return x.unsqueeze(0) if x.ndim == ndim else x


def _base_distance(query, support, metric):
    if metric == "euclidean":
        return euclidean_cost(query, support)
    if metric == "sqeuclidean":
        return squared_euclidean_cost(query, support)
    raise ValueError("metric must be 'euclidean' or 'sqeuclidean'")


def soft_dtw(cost, *, gamma=0.1):
    """soft-DTW paths for a cost matrix [B, T, U] (or [T, U])."""
    cost = _batched(cost, 2)
    d, pm = _value_and_grad(lambda x: _sdtw.soft_dtw(x, gamma), cost)
    return Alignment(d, pm, optimal_path(cost), cost)


def soft_dtw_per_view(query, support, *, gamma=0.1, metric="euclidean", angles=None):
    """soft-DTW (view-wise): each viewpoint of the query aligned separately.

    query [B, K, T, d] (or [K, T, d]), support [B, U, d] (or [U, d]).
    Returns a list of K Alignments; path cells are (t, u, k).
    """
    D = _base_distance(_batched(query, 3), _batched(support, 2), metric)
    out = []
    for k in range(D.shape[1]):
        a = soft_dtw(D[:, k], gamma=gamma)
        a.optimal_path = [[(t, u, k) for t, u in p] for p in a.optimal_path]
        a.angles = angles
        out.append(a)
    return out


def udtw(X, Y, sigma_x, sigma_y, *, gamma=1.0, beta=1.0):
    """uDTW paths. X [B, N, d], Y [B, M, d], sigma_x [B, N, 1], sigma_y [B, M, 1].

    The paths are evaluated on D * Sigma^dagger (uDTW Eq. 2); ``Sigma`` holds
    the pair-wise variances and ``Omega`` the penalty times beta (Eq. 3).
    """
    with torch.no_grad():
        DS, pen, Sigma = pairwise_matrices(X, Y, sigma_x, sigma_y, beta)
    d, pm = _value_and_grad(lambda x: udtw_from_cost(x, pen, gamma=gamma).distance, DS)
    with torch.no_grad():
        Omega = udtw_from_cost(DS, pen, gamma=gamma).penalty
    return Alignment(d, pm, optimal_path(DS), DS, Sigma=Sigma, Omega=Omega)


def fvm(query, support, *, gamma=0.1, metric="euclidean", angles=None):
    """FVM paths (query viewpoints only); query [B, K, T, d], support [B, U, d].

    At each step the viewpoint is free (JEANIE Eq. 13).
    """
    D = _base_distance(_batched(query, 3), _batched(support, 2), metric)
    d, pm = _value_and_grad(lambda x: fvm_query_only_1d(x, gamma), D)
    return Alignment(d, pm, optimal_path(D, max_shift=D.shape[1]), D, angles=angles)


def jeanie(query, support, *, gamma=0.1, max_shift=1, metric="euclidean", angles=None,
           start_view=None):
    """JEANIE paths (one viewpoint axis); query [B, K, T, d], support [B, U, d].

    ``max_shift`` is the iota-max shift. ``start_view`` restricts the paths to
    start at that viewpoint index (JEANIE Fig. 7 shows the optimal path for
    each possible starting position).
    """
    D = _base_distance(_batched(query, 3), _batched(support, 2), metric)
    Ds = D
    if start_view is not None:
        block = torch.zeros_like(D)
        block[:, :, 0, 0] = 1e4
        block[:, start_view, 0, 0] = 0
        Ds = D + block
    d, pm = _value_and_grad(lambda x: jeanie_1d_from_cost(x, gamma, max_shift), Ds)
    return Alignment(d, pm, optimal_path(D, max_shift=max_shift, start_view=start_view), D,
                     angles=angles)
