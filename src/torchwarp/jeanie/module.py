"""``nn.Module`` front ends for JEANIE and FVM."""

import torch.nn as nn

from .functional import _max_shifts, jeanie
from .fvm import _METRICS, fvm


class JEANIE(nn.Module):
    """Joint tEmporal and cAmera viewpoiNt alIgnmEnt (JEANIE); see
    ``torchwarp.functional.jeanie``.

    Args:
        gamma: SoftMin temperature (> 0).
        max_shift: iota-max shift. An int applies to every viewpoint axis; a
            pair ``(azimuth, altitude)`` sets the two axes separately.
        metric: Base distance: "euclidean" (default), "sqeuclidean" or "rbf".
        backend: "auto" (CUDA kernel when possible), "cuda" or "torch".

    Forward:
        query: [B, K, T, D] (one viewpoint axis) or [B, K1, K2, T, D] (two
            axes) viewpoint-augmented temporal blocks.
        support: [B, U, D] temporal blocks.
        return_accumulator: also return the differentiable DP tensor
            [B, K, T, U] / [B, K1, K2, T, U].

    Returns:
        distance [B] (and the accumulator if requested).
    """

    def __init__(self, gamma=0.1, max_shift=1, metric="euclidean", *, backend="auto"):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if metric not in _METRICS:
            raise ValueError("metric must be one of {}".format(tuple(_METRICS)))
        _max_shifts(max_shift, 1 if isinstance(max_shift, int) else len(max_shift))
        self.gamma = float(gamma)
        self.max_shift = max_shift if isinstance(max_shift, int) else tuple(max_shift)
        self.metric = metric
        self.backend = backend

    def forward(self, query, support, *, return_accumulator=False):
        return jeanie(query, support, gamma=self.gamma, max_shift=self.max_shift,
                      metric=self.metric, return_accumulator=return_accumulator,
                      backend=self.backend)

    def extra_repr(self):
        return "gamma={}, max_shift={}, metric={!r}, backend={!r}".format(
            self.gamma, self.max_shift, self.metric, self.backend)


class FVM(nn.Module):
    """Free Viewpoint Matching (JEANIE paper, Eq. 13); see
    ``torchwarp.functional.fvm``.

    Args:
        gamma: SoftMin temperature (> 0).
        metric: Base distance: "euclidean" (default), "sqeuclidean" or "rbf".
        backend: "auto" (CUDA kernel when possible), "cuda" or "torch".

    Forward:
        query: [B, K..., T, D] with one or more viewpoint dims.
        support: [B, U, D], or [B, K'..., U, D] with its own viewpoint dims.

    Returns:
        distance [B]
    """

    def __init__(self, gamma=0.1, metric="euclidean", *, backend="auto"):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if metric not in _METRICS:
            raise ValueError("metric must be one of {}".format(tuple(_METRICS)))
        self.gamma = float(gamma)
        self.metric = metric
        self.backend = backend

    def forward(self, query, support):
        return fvm(query, support, gamma=self.gamma, metric=self.metric, backend=self.backend)

    def extra_repr(self):
        return "gamma={}, metric={!r}, backend={!r}".format(self.gamma, self.metric, self.backend)
