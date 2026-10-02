"""Fast JEANIE temporal-viewpoint alignment and soft-DTW.

Same mathematics as the reference implementation (github.com/LeiWangR/JEANIE;
Algorithm 1 of the paper and its two-axis extension). Every entry point accepts either a
single cost tensor or a batch with one extra leading dimension. The dynamic
program runs as a fused CUDA/HIP kernel or as a vectorised anti-diagonal
PyTorch loop, wrapped as PyTorch custom ops (see ``_ops``): analytic
first-order backward, exact higher-order gradients, a differentiable
accumulator, and support for torch.compile, torch.func and autocast.
"""

from typing import Optional

import torch

from . import _ops

_BACKENDS = ("auto", "cuda", "torch")
_METRICS = {"sqeuclidean": 0, "euclidean": 1}


def softmin(x: torch.Tensor, gamma: float, dim: Optional[int] = None) -> torch.Tensor:
    """Differentiable soft minimum -gamma * logsumexp(-x / gamma)."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if dim is None:
        x = x.reshape(-1)
        dim = 0
    return -gamma * torch.logsumexp(-x / gamma, dim=dim)


def _check_common(gamma, max_shift_1, max_shift_2, backend):
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if max_shift_1 < 0 or max_shift_2 < 0:
        raise ValueError("max shifts must be >= 0")
    if backend not in _BACKENDS:
        raise ValueError("backend must be one of {}".format(_BACKENDS))


def jeanie_dp(cost, gamma=0.1, max_shift_1=1, max_shift_2=1, backend="auto",
              return_accumulator=False):
    """Batched JEANIE on a [B, K1, K2, T, U] cost tensor.

    Returns:
        (distance [B], accumulator [B, K1, K2, T, U] or None). The
        accumulator is only materialised when ``return_accumulator`` is set;
        it is differentiable, like in the reference implementation.
    """
    if cost.ndim != 5:
        raise ValueError("cost must have shape [B, K1, K2, T, U]")
    if not cost.is_floating_point():
        raise TypeError("cost must be a floating-point tensor")
    _check_common(gamma, max_shift_1, max_shift_2, backend)
    B, K1, K2, T, U = cost.shape
    if min(K1, K2, T, U) == 0:
        raise ValueError("cost dimensions must be non-empty")
    if cost.dtype not in (torch.float32, torch.float64):
        cost = cost.float()

    # Shifts beyond K-1 reach every viewpoint anyway.
    s1 = min(int(max_shift_1), K1 - 1)
    s2 = min(int(max_shift_2), K2 - 1)

    # The temporal moves are symmetric in (t, u); keeping T <= U bounds the
    # diagonal length (and the kernel's shared memory) by T.
    transposed = T > U
    if transposed:
        cost = cost.transpose(3, 4)

    out, _, R = _ops.dp_apply(cost, s1, s2, float(gamma), backend, bool(return_accumulator))
    if not return_accumulator:
        return out, None
    return out, (R.transpose(3, 4) if transposed else R)


def jeanie_dp_features(query, support, gamma=0.1, max_shift_1=1, max_shift_2=1,
                       metric="euclidean", backend="auto", return_accumulator=False):
    """Batched JEANIE straight from features.

    Args:
        query: [B, K1, K2, T, D] viewpoint-augmented query blocks.
        support: [B, U, D] support blocks.
        metric: "euclidean", "sqeuclidean" or "rbf".

    Returns:
        (distance [B], accumulator [B, K1, K2, T, U] or None); the
        accumulator is only built when requested and is differentiable.

    For the Euclidean metrics the cost tensor is never materialised on CUDA
    (fused op); "rbf" goes through ``jeanie_dp(rbf_cost(query, support))``.
    """
    if query.ndim != 5 or support.ndim != 3:
        raise ValueError("query must be [B,K1,K2,T,D] and support [B,U,D]")
    if query.shape[0] != support.shape[0] or query.shape[-1] != support.shape[-1]:
        raise ValueError("query/support batch or feature dimensions do not match")
    if metric not in ("euclidean", "sqeuclidean", "rbf"):
        raise ValueError("metric must be 'euclidean', 'sqeuclidean' or 'rbf'")
    _check_common(gamma, max_shift_1, max_shift_2, backend)
    B, K1, K2, T, D = query.shape
    U = support.shape[1]
    if min(K1, K2, T, U) == 0:
        raise ValueError("query/support dimensions must be non-empty")

    if metric == "rbf":
        from .distances import rbf_cost

        cost = rbf_cost(query.reshape(B, K1 * K2, T, D), support).reshape(B, K1, K2, T, U)
        return jeanie_dp(cost, gamma, max_shift_1, max_shift_2, backend, return_accumulator)

    dtype = torch.promote_types(query.dtype, support.dtype)
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    s1 = min(int(max_shift_1), K1 - 1)
    s2 = min(int(max_shift_2), K2 - 1)
    out, _, R, _, _, _ = _ops.features_apply(
        query.to(dtype), support.to(dtype).contiguous(), s1, s2, float(gamma),
        _METRICS[metric], backend, bool(return_accumulator))
    return out, (R if return_accumulator else None)


def jeanie_1d_from_features(query, support, gamma=0.1, max_shift=1,
                            metric="euclidean", return_accumulator=False, backend="auto"):
    """JEANIE (one viewpoint axis) from features.

    Args:
        query: [K, T, D] or batched [B, K, T, D].
        support: [U, D] or batched [B, U, D].
        metric: "euclidean" (paper default), "sqeuclidean" or "rbf".
    """
    single = query.ndim == 3
    if single:
        query, support = query.unsqueeze(0), support.unsqueeze(0)
    if query.ndim != 4 or support.ndim != 3:
        raise ValueError("query must be [K,T,D] / [B,K,T,D] and support [U,D] / [B,U,D]")
    out, R = jeanie_dp_features(query.unsqueeze(2), support, gamma, max_shift, 0,
                                metric, backend, return_accumulator)
    if not return_accumulator:
        return out[0] if single else out
    R = R[:, :, 0]
    if single:
        out, R = out[0], R[0]
    return (out, R) if return_accumulator else out


def jeanie_2d_from_features(query, support, gamma=0.1, max_shift_az=1, max_shift_alt=1,
                            metric="euclidean", return_accumulator=False, backend="auto"):
    """JEANIE on a two-axis viewpoint grid from features.

    Args:
        query: [K1, K2, T, D] or batched [B, K1, K2, T, D].
        support: [U, D] or batched [B, U, D].
    """
    single = query.ndim == 4
    if single:
        query, support = query.unsqueeze(0), support.unsqueeze(0)
    if query.ndim != 5 or support.ndim != 3:
        raise ValueError("query must be [K1,K2,T,D] / [B,K1,K2,T,D] and support [U,D] / [B,U,D]")
    out, R = jeanie_dp_features(query, support, gamma, max_shift_az, max_shift_alt,
                                metric, backend, return_accumulator)
    if not return_accumulator:
        return out[0] if single else out
    if single:
        out, R = out[0], R[0]
    return (out, R) if return_accumulator else out


def _batchify(cost, unbatched_ndim):
    if cost.ndim == unbatched_ndim:
        return cost.unsqueeze(0), True
    if cost.ndim == unbatched_ndim + 1:
        return cost, False
    raise ValueError(
        "cost must have {} dims (or {} with a leading batch dim)".format(
            unbatched_ndim, unbatched_ndim + 1
        )
    )


def soft_dtw(cost: torch.Tensor, gamma: float = 0.1, backend: str = "auto") -> torch.Tensor:
    """soft-DTW for a [T, U] cost (scalar) or a [B, T, U] batch ([B])."""
    if not cost.is_floating_point():
        raise TypeError("cost must be a floating-point tensor")
    c, single = _batchify(cost, 2)
    out, _ = jeanie_dp(c[:, None, None], gamma, 0, 0, backend)
    return out[0] if single else out


def jeanie_1d_from_cost(
    cost: torch.Tensor,
    gamma: float = 0.1,
    max_shift: int = 1,
    return_accumulator: bool = False,
    backend: str = "auto",
):
    """JEANIE for one viewpoint axis (Algorithm 1).

    Args:
        cost: [K, T, U] or batched [B, K, T, U].
        gamma: Positive soft-min temperature.
        max_shift: Viewpoint smoothness (iota-max shift).
        return_accumulator: Also return the (differentiable) DP tensor.
        backend: "auto", "cuda" or "torch".

    Returns:
        Distance (scalar, or [B]), and optionally the accumulator.
    """
    if not cost.is_floating_point():
        raise TypeError("cost must be a floating-point tensor")
    if max_shift < 0:
        raise ValueError("max_shift must be >= 0")
    c, single = _batchify(cost, 3)
    out, R = jeanie_dp(c[:, :, None], gamma, max_shift, 0, backend, return_accumulator)
    if not return_accumulator:
        return out[0] if single else out
    R = R[:, :, 0]
    if single:
        out, R = out[0], R[0]
    return (out, R) if return_accumulator else out


def jeanie_2d_from_cost(
    cost: torch.Tensor,
    gamma: float = 0.1,
    max_shift_az: int = 1,
    max_shift_alt: int = 1,
    return_accumulator: bool = False,
    backend: str = "auto",
):
    """JEANIE on a two-axis viewpoint grid.

    Args:
        cost: [K1, K2, T, U] or batched [B, K1, K2, T, U].
        gamma: Positive soft-min temperature.
        max_shift_az: Maximum predecessor shift on viewpoint axis 1.
        max_shift_alt: Maximum predecessor shift on viewpoint axis 2.
        return_accumulator: Also return the (differentiable) DP tensor.
        backend: "auto", "cuda" or "torch".
    """
    if not cost.is_floating_point():
        raise TypeError("cost must be a floating-point tensor")
    if max_shift_az < 0 or max_shift_alt < 0:
        raise ValueError("max shifts must be >= 0")
    c, single = _batchify(cost, 4)
    out, R = jeanie_dp(c, gamma, max_shift_az, max_shift_alt, backend, return_accumulator)
    if not return_accumulator:
        return out[0] if single else out
    if single:
        out, R = out[0], R[0]
    return (out, R) if return_accumulator else out
