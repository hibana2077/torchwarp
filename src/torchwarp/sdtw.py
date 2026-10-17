"""soft-DTW (Cuturi & Blondel, ICML 2017)."""

from typing import Optional

import torch
import torch.nn as nn

from .jeanie.functional import _jeanie_dp
from .udtw.functional import _BACKENDS, udtw, udtw_from_cost


def softmin(x: torch.Tensor, gamma: float, dim: Optional[int] = None) -> torch.Tensor:
    """Differentiable soft minimum -gamma * logsumexp(-x / gamma)."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if dim is None:
        x = x.reshape(-1)
        dim = 0
    return -gamma * torch.logsumexp(-x / gamma, dim=dim)


def softmin_weights(values: torch.Tensor, gamma: float, dim: int = 0) -> torch.Tensor:
    """Weights of the soft minimum, softmax(-values / gamma) over ``dim``."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    return torch.softmax(-values / gamma, dim=dim)


def soft_dtw_from_cost(cost: torch.Tensor, *, gamma: float = 1.0,
                       bandwidth: Optional[float] = None, backend: str = "auto") -> torch.Tensor:
    """soft-DTW on a precomputed cost [B, T, U]; returns [B].

    Args:
        cost: [B, T, U] cost matrices.
        gamma: SoftMin temperature (> 0).
        bandwidth: Sakoe-Chiba band; None or 0 disables pruning.
        backend: "auto" (CUDA kernel when possible), "cuda" or "torch".
    """
    if not cost.is_floating_point():
        raise TypeError("cost must be a floating-point tensor")
    if cost.ndim != 3:
        raise ValueError("cost must have shape [B, T, U]")
    if bandwidth:
        return udtw_from_cost(cost, torch.zeros_like(cost), gamma=gamma, bandwidth=bandwidth,
                              backend=backend).distance
    out, _ = _jeanie_dp(cost[:, None, None], gamma, 0, 0, backend)
    return out


def soft_dtw(X: torch.Tensor, Y: torch.Tensor, *, gamma: float = 1.0, normalize: bool = False,
             bandwidth: Optional[float] = None, dist_func=None,
             backend: str = "auto") -> torch.Tensor:
    """soft-DTW between two batches of sequences.

    Args:
        X, Y: [B,N,D], [B,M,D] sequences.
        gamma: SoftMin temperature (> 0).
        normalize: If True, return the soft-DTW divergence
            d(X,Y) - 0.5 [d(X,X) + d(Y,Y)].
        bandwidth: Sakoe-Chiba band; None or 0 disables pruning.
        dist_func: Optional function (X, Y) -> cost [B,N,M]. Default: squared
            Euclidean distance; the CUDA kernel computes it without storing
            the [B,N,M] matrix.
        backend: "auto" (CUDA kernel when possible), "cuda" or "torch".

    Returns:
        distance [B]
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if bandwidth is not None and bandwidth < 0:
        raise ValueError("bandwidth must be None or >= 0")
    if backend not in _BACKENDS:
        raise ValueError("backend must be one of {}".format(_BACKENDS))
    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError("X and Y must have shape [batch, time, feature]")

    def run(A, B):
        if dist_func is None:
            # soft-DTW is uDTW with sigma = 1 (so Sigma = 1) and beta = 0
            ones_a = A.new_ones(A.shape[0], A.shape[1], 1)
            ones_b = B.new_ones(B.shape[0], B.shape[1], 1)
            return udtw(A, B, ones_a, ones_b, beta=0.0, gamma=gamma, bandwidth=bandwidth,
                        backend=backend).distance
        cost = dist_func(A, B)
        return udtw_from_cost(cost, torch.zeros_like(cost), gamma=gamma, bandwidth=bandwidth,
                              backend=backend).distance

    d = run(X, Y)
    if not normalize:
        return d
    return d - 0.5 * (run(X, X) + run(Y, Y))


class SoftDTW(nn.Module):
    """soft-DTW module; see ``torchwarp.functional.soft_dtw``.

    Args:
        gamma: SoftMin temperature (> 0).
        normalize: If True, return the soft-DTW divergence
            d(X,Y) - 0.5 [d(X,X) + d(Y,Y)].
        bandwidth: Sakoe-Chiba band; None or 0 disables pruning.
        dist_func: Optional function (X, Y) -> cost [B,N,M]. Default: squared
            Euclidean distance; the CUDA kernel computes it without storing
            the [B,N,M] matrix.
        backend: "auto", "cuda" or "torch".

    Forward:
        X, Y: [B,N,D], [B,M,D]

    Returns:
        distance [B]
    """

    def __init__(self, gamma=1.0, normalize=False, bandwidth=None, dist_func=None, *,
                 backend="auto"):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if bandwidth is not None and bandwidth < 0:
            raise ValueError("bandwidth must be None or >= 0")
        if backend not in _BACKENDS:
            raise ValueError("backend must be one of {}".format(_BACKENDS))
        self.gamma = float(gamma)
        self.normalize = bool(normalize)
        self.bandwidth = None if bandwidth is None else float(bandwidth)
        self.dist_func = dist_func
        self.backend = backend

    def forward(self, X, Y):
        return soft_dtw(X, Y, gamma=self.gamma, normalize=self.normalize,
                        bandwidth=self.bandwidth, dist_func=self.dist_func, backend=self.backend)

    def extra_repr(self):
        return "gamma={}, normalize={}, bandwidth={}, backend={!r}".format(
            self.gamma, self.normalize, self.bandwidth, self.backend)
