"""Fast uncertainty-DTW: functional interface.

Implements the uDTW formulation of Wang & Koniusz (ECCV 2022):
  * uncertainty-weighted path cost (Eq. 2 / Eq. 16)
  * uncertainty penalty under the same soft path distribution (Eq. 3)
  * additive pairwise variance (Eq. 8)
  * squared-Euclidean base distance (Eqs. 14-16)

The dynamic program runs as a fused CUDA/HIP kernel or as a vectorised
anti-diagonal PyTorch loop, wrapped as PyTorch custom ops (see ``_ops``):
analytic first-order backward, exact higher-order gradients, and support
for torch.compile, torch.func and autocast.
"""

from typing import NamedTuple

import torch

from . import _ops

_BACKENDS = ("auto", "cuda", "torch")


class UDTWOutput(NamedTuple):
    """Result of uDTW: ``distance`` [B] and the beta-weighted ``penalty`` [B].

    A tuple, so ``d, omega = ...`` also works.
    """

    distance: torch.Tensor
    penalty: torch.Tensor


def _check_sequence_inputs(X, Y, Sigma_X, Sigma_Y):
    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError("X and Y must have shape [batch, time, feature]")
    if Sigma_X.ndim != 3 or Sigma_Y.ndim != 3:
        raise ValueError("the uncertainties sigma must have shape [batch, time, 1]")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same batch size")
    if X.shape[2] != Y.shape[2]:
        raise ValueError("X and Y must have the same feature dimension")
    if Sigma_X.shape[:2] != X.shape[:2] or Sigma_Y.shape[:2] != Y.shape[:2]:
        raise ValueError("uncertainty tensors must match batch/time dimensions")
    if Sigma_X.shape[2] != 1 or Sigma_Y.shape[2] != 1:
        raise ValueError("the uncertainties sigma must have shape [batch, time, 1]")
    if not X.is_floating_point() or not Y.is_floating_point():
        raise TypeError("X and Y must be floating-point tensors")
    if not Sigma_X.is_floating_point() or not Sigma_Y.is_floating_point():
        raise TypeError("the uncertainties sigma must be floating-point tensors")


def _check_hyperparameters(gamma, beta, bandwidth):
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if beta < 0:
        raise ValueError("beta must be >= 0")
    if bandwidth is not None and bandwidth < 0:
        raise ValueError("bandwidth must be None or >= 0")


def _squared_euclidean(X, Y, method="gemm"):
    """Pairwise squared distances [B,N,M] between X [B,N,D] and Y [B,M,D].

    ``"gemm"`` uses ||x||^2 + ||y||^2 - 2 x.y (one batched matmul, low
    memory); ``"diff"`` materialises x - y.
    """
    if method == "diff":
        return (X.unsqueeze(2) - Y.unsqueeze(1)).pow(2).sum(dim=3)
    if method != "gemm":
        raise ValueError("method must be 'gemm' or 'diff'")
    x2 = X.pow(2).sum(-1, keepdim=True)
    y2 = Y.pow(2).sum(-1).unsqueeze(1)
    return torch.baddbmm(x2 + y2, X, Y.transpose(1, 2), alpha=-2.0).clamp_min(0)


def pairwise_matrices(X, Y, Sigma_X, Sigma_Y, beta=1.0, eps=1e-8, *, method="gemm"):
    """Uncertainty-weighted cost, beta-weighted log-variance penalty, variance.

    Eq. (8):  Sigma_ij = 0.5 * (sigma_i^2 + sigma'_j^2)
    Eq. (16): cost_ij = ||x_i - y_j||^2 / Sigma_ij, penalty_ij = beta * log Sigma_ij
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if beta < 0:
        raise ValueError("beta must be >= 0")
    variance = (0.5 * (Sigma_X.pow(2) + Sigma_Y.pow(2).transpose(1, 2))).clamp_min(eps)
    weighted_cost = _squared_euclidean(X, Y, method) / variance
    weighted_penalty = beta * torch.log(variance)
    return weighted_cost, weighted_penalty, variance


def _check_band_reached(distance, band):
    # Data-dependent check: skipped while torch.compile traces the graph
    # (an unreachable end cell then shows up as +inf instead of an error).
    if band > 0 and not torch.compiler.is_compiling():
        if not bool(torch.isfinite(distance).all()):
            raise ValueError(
                "No valid DTW path reaches the final cell under the given bandwidth"
            )


def udtw_from_cost(cost, penalty, *, gamma=1.0, bandwidth=None, backend="auto"):
    """Run the uDTW dynamic program on precomputed matrices.

    Args:
        cost: [B,N,M] uncertainty-weighted cost, e.g. D / Sigma.
        penalty: [B,N,M] per-cell penalty, e.g. beta * log Sigma.
        gamma: SoftMin temperature (> 0).
        bandwidth: Sakoe-Chiba band; None or 0 disables pruning.
        backend: "auto" (CUDA kernel when possible), "cuda" or "torch".

    Returns:
        UDTWOutput(distance [B], penalty [B]).

    Use this directly for custom cost/variance constructions such as the
    jointly generated variance of Eq. (9).
    """
    if cost.ndim != 3 or penalty.ndim != 3:
        raise ValueError("cost and penalty must have shape [B,N,M]")
    if cost.shape != penalty.shape:
        raise ValueError("cost and penalty must have identical shapes")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if bandwidth is not None and bandwidth < 0:
        raise ValueError("bandwidth must be None or >= 0")
    if backend not in _BACKENDS:
        raise ValueError("backend must be one of {}".format(_BACKENDS))
    if cost.shape[1] == 0 or cost.shape[2] == 0:
        raise ValueError("sequence lengths must be non-zero")
    if cost.dtype not in (torch.float32, torch.float64):
        cost, penalty = cost.float(), penalty.float()

    # The recurrence and the band are symmetric in (i, j); keeping N <= M
    # bounds the diagonal length (and the kernel's shared memory) by N.
    if cost.shape[1] > cost.shape[2]:
        cost = cost.transpose(1, 2)
        penalty = penalty.transpose(1, 2)

    band = 0.0 if bandwidth is None else float(bandwidth)
    distance, pen, _, _ = _ops.dp_apply(cost, penalty, float(gamma), band, backend)
    _check_band_reached(distance, band)
    return UDTWOutput(distance, pen)


def udtw_from_features(X, Y, sigma_x, sigma_y, *, beta=1.0, gamma=1.0, bandwidth=None,
                       eps=1e-8, backend="auto"):
    """uDTW with the cost and penalty built inside the kernel (Eqs. 8, 16).

    X [B,N,D], Y [B,M,D]; sigma_x [B,N,1], sigma_y [B,M,1] positive standard
    deviations. Same result as ``udtw_from_cost(*pairwise_matrices(...)[:2])``
    but without materialising the [B,N,M] cost/penalty/variance tensors.

    Returns:
        UDTWOutput(distance [B], penalty [B]).
    """
    _check_sequence_inputs(X, Y, sigma_x, sigma_y)
    _check_hyperparameters(gamma, beta, bandwidth)
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if backend not in _BACKENDS:
        raise ValueError("backend must be one of {}".format(_BACKENDS))
    if X.shape[1] == 0 or Y.shape[1] == 0:
        raise ValueError("sequence lengths must be non-zero")
    if X.shape[1] > Y.shape[1]:  # symmetric problem; keep N <= M
        X, Y, sigma_x, sigma_y = Y, X, sigma_y, sigma_x
    dtype = X.dtype if X.dtype in (torch.float32, torch.float64) else torch.float32
    X, Y = X.to(dtype), Y.to(dtype)
    if Y is X:
        # d(X, X) for normalize=True: torch.compile cannot trace an
        # autograd.Function that receives the same tensor twice.
        Y = X.view_as(X)
    vx = sigma_x.squeeze(-1).to(dtype).pow(2)
    vy = sigma_y.squeeze(-1).to(dtype).pow(2)
    band = 0.0 if bandwidth is None else float(bandwidth)
    distance, pen, *_ = _ops.fused_apply(
        X, Y, vx, vy, float(gamma), float(beta), float(eps), band, backend)
    _check_band_reached(distance, band)
    return UDTWOutput(distance, pen)
