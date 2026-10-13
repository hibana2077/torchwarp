"""``nn.Module`` front end for uDTW."""

import torch.nn as nn

from .functional import (
    _BACKENDS,
    UDTWOutput,
    _check_hyperparameters,
    _check_sequence_inputs,
    pairwise_matrices,
    udtw_from_cost,
    udtw_from_features,
)


class UDTW(nn.Module):
    """Uncertainty-DTW module.

    Args:
        use_cuda: Kept for backward compatibility; the device of the inputs
            decides where the computation runs.
        gamma: Soft-DTW relaxation temperature.
        normalize: If True, return d(X,Y) - 0.5[d(X,X)+d(Y,Y)] for both the
            distance and the penalty.
        bandwidth: Optional Sakoe-Chiba bandwidth. None or 0 disables pruning.
        backend: "auto", "cuda" or "torch".
        method: "gemm" (default: cost built inside the fused op) or "diff"
            (materialised [B,N,M] matrices).

    The first four arguments are those of the official uDTW code; the
    others are keyword-only.

    Forward:
        X, Y: [B,N,D], [B,M,D]
        Sigma_X, Sigma_Y: [B,N,1], [B,M,1] positive standard deviations.
        beta: penalty coefficient from Eq. (15).

    Returns:
        UDTWOutput(distance [B], penalty [B]); the penalty is beta-weighted.
    """

    def __init__(self, use_cuda=False, gamma=1.0, normalize=False,
                 bandwidth=None, *, backend="auto", method="gemm"):
        super().__init__()
        self.use_cuda = bool(use_cuda)
        self.gamma = float(gamma)
        self.normalize = bool(normalize)
        self.bandwidth = None if bandwidth is None else float(bandwidth)
        if backend not in _BACKENDS:
            raise ValueError("backend must be one of {}".format(_BACKENDS))
        if method not in ("gemm", "diff"):
            raise ValueError("method must be 'gemm' or 'diff'")
        self.backend = backend
        self.method = method
        _check_hyperparameters(self.gamma, 0.0, self.bandwidth)

    def _dp(self, A, B, SA, SB, beta):
        kw = dict(gamma=self.gamma, bandwidth=self.bandwidth, backend=self.backend)
        if self.method == "gemm":
            return udtw_from_features(A, B, SA, SB, beta=beta, **kw)
        cost, pen, _ = pairwise_matrices(A, B, SA, SB, beta, method=self.method)
        return udtw_from_cost(cost, pen, **kw)

    def forward(self, X, Y, Sigma_X, Sigma_Y, beta=1.0):
        _check_sequence_inputs(X, Y, Sigma_X, Sigma_Y)
        _check_hyperparameters(self.gamma, beta, self.bandwidth)

        out_xy, pen_xy = self._dp(X, Y, Sigma_X, Sigma_Y, beta)
        if not self.normalize:
            return UDTWOutput(out_xy, pen_xy)

        out_xx, pen_xx = self._dp(X, X, Sigma_X, Sigma_X, beta)
        out_yy, pen_yy = self._dp(Y, Y, Sigma_Y, Sigma_Y, beta)
        return UDTWOutput(out_xy - 0.5 * (out_xx + out_yy), pen_xy - 0.5 * (pen_xx + pen_yy))


# Name used by the official uDTW code and earlier torchwarp versions.
uDTW = UDTW
