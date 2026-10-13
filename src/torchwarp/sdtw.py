"""soft-DTW (Cuturi & Blondel, ICML 2017).

``soft_dtw`` and ``softmin`` have the signatures of the official JEANIE
code, ``softmin_weights`` that of the official uDTW code, and ``SoftDTW``
the arguments of pytorch-softdtw-cuda's ``SoftDTW``.
"""

from typing import Optional

import torch
import torch.nn as nn

from .jeanie.functional import _batchify, _jeanie_dp
from .udtw.functional import _BACKENDS, udtw_from_cost, udtw_from_features


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


def soft_dtw(cost: torch.Tensor, gamma: float = 0.1, *, backend: str = "auto") -> torch.Tensor:
    """soft-DTW for a [T, U] cost (scalar) or a [B, T, U] batch ([B])."""
    if not cost.is_floating_point():
        raise TypeError("cost must be a floating-point tensor")
    c, single = _batchify(cost, 2)
    out, _ = _jeanie_dp(c[:, None, None], gamma, 0, 0, backend)
    return out[0] if single else out


class SoftDTW(nn.Module):
    """soft-DTW module, with the arguments of pytorch-softdtw-cuda's ``SoftDTW``.

    Args:
        use_cuda: Ignored; the device of the inputs decides where the
            computation runs.
        gamma: SoftMin temperature (> 0).
        normalize: If True, return the soft-DTW divergence
            d(X,Y) - 0.5 [d(X,X) + d(Y,Y)].
        bandwidth: Sakoe-Chiba bandwidth. None or 0 disables pruning.
        dist_func: Optional function (X, Y) -> cost [B,N,M]. Default: squared
            Euclidean distance; the CUDA kernel computes it without storing
            the [B,N,M] matrix.
        backend: "auto", "cuda" or "torch" (keyword-only).

    Forward:
        X, Y: [B,N,D], [B,M,D]

    Returns:
        distance [B]
    """

    def __init__(self, use_cuda=False, gamma=1.0, normalize=False, bandwidth=None,
                 dist_func=None, *, backend="auto"):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if bandwidth is not None and bandwidth < 0:
            raise ValueError("bandwidth must be None or >= 0")
        if backend not in _BACKENDS:
            raise ValueError("backend must be one of {}".format(_BACKENDS))
        self.use_cuda = bool(use_cuda)
        self.gamma = float(gamma)
        self.normalize = bool(normalize)
        self.bandwidth = None if bandwidth is None else float(bandwidth)
        self.dist_func = dist_func
        self.backend = backend

    def _dp(self, X, Y):
        kw = dict(gamma=self.gamma, bandwidth=self.bandwidth, backend=self.backend)
        if self.dist_func is None:
            # soft-DTW is uDTW with sigma = 1 (so Sigma = 1) and beta = 0
            ones_x = X.new_ones(X.shape[0], X.shape[1], 1)
            ones_y = Y.new_ones(Y.shape[0], Y.shape[1], 1)
            return udtw_from_features(X, Y, ones_x, ones_y, beta=0.0, **kw).distance
        cost = self.dist_func(X, Y)
        return udtw_from_cost(cost, torch.zeros_like(cost), **kw).distance

    def forward(self, X, Y):
        d = self._dp(X, Y)
        if not self.normalize:
            return d
        return d - 0.5 * (self._dp(X, X) + self._dp(Y, Y))

    def extra_repr(self):
        return "gamma={}, normalize={}, bandwidth={}, backend={!r}".format(
            self.gamma, self.normalize, self.bandwidth, self.backend)
