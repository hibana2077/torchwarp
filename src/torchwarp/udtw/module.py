"""``nn.Module`` front end for uDTW."""

import torch.nn as nn

from .functional import (
    _BACKENDS,
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
        distance: "gemm" (default: cost built inside the fused op) or "diff"
            (materialised [B,N,M] matrices).

    Forward:
        X, Y: [B,N,D], [B,M,D]
        Sigma_X, Sigma_Y: [B,N,1], [B,M,1] positive standard deviations.
        beta: penalty coefficient from Eq. (15).

    Returns:
        (distance [B], beta_weighted_penalty [B])
    """

    def __init__(self, use_cuda=False, gamma=1.0, normalize=False,
                 bandwidth=None, backend="auto", distance="gemm"):
        super().__init__()
        self.use_cuda = bool(use_cuda)
        self.gamma = float(gamma)
        self.normalize = bool(normalize)
        self.bandwidth = None if bandwidth is None else float(bandwidth)
        if backend not in _BACKENDS:
            raise ValueError("backend must be one of {}".format(_BACKENDS))
        if distance not in ("gemm", "diff"):
            raise ValueError("distance must be 'gemm' or 'diff'")
        self.backend = backend
        self.distance = distance
        _check_hyperparameters(self.gamma, 0.0, self.bandwidth)

    def _dp(self, A, B, SA, SB, beta):
        if self.distance == "gemm":
            return udtw_from_features(A, B, SA, SB, beta, self.gamma, self.bandwidth,
                                      backend=self.backend)
        cost, pen, _ = pairwise_matrices(A, B, SA, SB, beta, method=self.distance)
        return udtw_from_cost(cost, pen, self.gamma, self.bandwidth, self.backend)

    def forward(self, X, Y, Sigma_X, Sigma_Y, beta=1.0):
        _check_sequence_inputs(X, Y, Sigma_X, Sigma_Y)
        _check_hyperparameters(self.gamma, beta, self.bandwidth)

        out_xy, pen_xy = self._dp(X, Y, Sigma_X, Sigma_Y, beta)
        if not self.normalize:
            return out_xy, pen_xy

        out_xx, pen_xx = self._dp(X, X, Sigma_X, Sigma_X, beta)
        out_yy, pen_yy = self._dp(Y, Y, Sigma_Y, Sigma_Y, beta)
        return out_xy - 0.5 * (out_xx + out_yy), pen_xy - 0.5 * (pen_xx + pen_yy)


# Name used by the official uDTW code and earlier torchwarp versions.
uDTW = UDTW
