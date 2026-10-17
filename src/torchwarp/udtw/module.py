"""``nn.Module`` front end for uDTW."""

import torch.nn as nn

from .functional import _BACKENDS, _check_hyperparameters, udtw


class UDTW(nn.Module):
    """Uncertainty-DTW module; see ``torchwarp.functional.udtw``.

    Args:
        gamma: SoftMin temperature (> 0).
        beta: Penalty weight (>= 0).
        normalize: If True, return d(X,Y) - 0.5 [d(X,X) + d(Y,Y)] for both the
            distance and the penalty.
        bandwidth: Sakoe-Chiba band; None or 0 disables pruning.
        method: "gemm" (default: cost built inside the fused kernel) or
            "diff" (materialised [B,N,M] matrices).
        backend: "auto", "cuda" or "torch".

    Forward:
        X, Y: [B,N,D], [B,M,D]
        sigma_x, sigma_y: [B,N,1], [B,M,1] positive standard deviations.

    Returns:
        UDTWOutput(distance [B], penalty [B]); the penalty is beta-weighted.
    """

    def __init__(self, gamma=1.0, beta=1.0, normalize=False, bandwidth=None, *,
                 method="gemm", backend="auto"):
        super().__init__()
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.normalize = bool(normalize)
        self.bandwidth = None if bandwidth is None else float(bandwidth)
        if backend not in _BACKENDS:
            raise ValueError("backend must be one of {}".format(_BACKENDS))
        if method not in ("gemm", "diff"):
            raise ValueError("method must be 'gemm' or 'diff'")
        self.method = method
        self.backend = backend
        _check_hyperparameters(self.gamma, self.beta, self.bandwidth)

    def forward(self, X, Y, sigma_x, sigma_y):
        return udtw(X, Y, sigma_x, sigma_y, beta=self.beta, gamma=self.gamma,
                    normalize=self.normalize, bandwidth=self.bandwidth, method=self.method,
                    backend=self.backend)

    def extra_repr(self):
        return "gamma={}, beta={}, normalize={}, bandwidth={}, method={!r}, backend={!r}".format(
            self.gamma, self.beta, self.normalize, self.bandwidth, self.method, self.backend)
