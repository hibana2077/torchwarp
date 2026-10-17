"""Fast JEANIE temporal-viewpoint alignment: functional interface.

Implements JEANIE (Wang et al., IJCV 2024; Algorithm 1 and its two-axis
extension). Every entry point takes a batch (leading dimension B). The dynamic
program runs as a fused CUDA/HIP kernel or as a vectorised anti-diagonal
PyTorch loop, wrapped as PyTorch custom ops (see ``_ops``): analytic
first-order backward, exact higher-order gradients, a differentiable
accumulator, and support for torch.compile, torch.func and autocast.
"""

import torch

from . import _ops

_BACKENDS = ("auto", "cuda", "torch")
_METRICS = {"sqeuclidean": 0, "euclidean": 1}


def _check_common(gamma, max_shift_1, max_shift_2, backend):
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if max_shift_1 < 0 or max_shift_2 < 0:
        raise ValueError("max shifts must be >= 0")
    if backend not in _BACKENDS:
        raise ValueError("backend must be one of {}".format(_BACKENDS))


def _jeanie_dp(cost, gamma=0.1, max_shift_1=1, max_shift_2=1, backend="auto",
              return_accumulator=False):
    """Batched JEANIE on a [B, K1, K2, T, U] cost tensor.

    Returns:
        (distance [B], accumulator [B, K1, K2, T, U] or None). The
        accumulator is only materialised when ``return_accumulator`` is set;
        it is differentiable.
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


def _jeanie_dp_features(query, support, gamma=0.1, max_shift_1=1, max_shift_2=1,
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
    (fused op); "rbf" goes through ``_jeanie_dp(rbf_cost(query, support))``.
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
        return _jeanie_dp(cost, gamma, max_shift_1, max_shift_2, backend, return_accumulator)

    dtype = torch.promote_types(query.dtype, support.dtype)
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    s1 = min(int(max_shift_1), K1 - 1)
    s2 = min(int(max_shift_2), K2 - 1)
    out, _, R, _, _, _ = _ops.features_apply(
        query.to(dtype), support.to(dtype).contiguous(), s1, s2, float(gamma),
        _METRICS[metric], backend, bool(return_accumulator))
    return out, (R if return_accumulator else None)


def _max_shifts(max_shift, axes):
    """iota-max shift per viewpoint axis: an int applies to every axis."""
    if isinstance(max_shift, int):
        shifts = (max_shift,) * axes
    else:
        shifts = tuple(int(m) for m in max_shift)
        if len(shifts) != axes:
            raise ValueError("max_shift must be an int or a tuple of {} ints for {} viewpoint "
                             "axis/axes".format(axes, axes))
    if min(shifts) < 0:
        raise ValueError("max_shift must be >= 0")
    return shifts + (0,) * (2 - axes)


def jeanie(query, support, *, gamma=0.1, max_shift=1, metric="euclidean",
           return_accumulator=False, backend="auto"):
    """JEANIE between viewpoint-augmented queries and supports.

    Args:
        query: [B, K, T, D] (one viewpoint axis) or [B, K1, K2, T, D] (two
            axes) temporal blocks.
        support: [B, U, D] temporal blocks.
        gamma: SoftMin temperature (> 0).
        max_shift: iota-max shift. An int applies to every viewpoint axis;
            a pair ``(azimuth, altitude)`` sets the two axes separately.
        metric: Base distance: "euclidean" (default), "sqeuclidean" or "rbf".
        return_accumulator: Also return the differentiable DP tensor
            [B, K, T, U] / [B, K1, K2, T, U].
        backend: "auto" (CUDA kernel when possible), "cuda" or "torch".

    Returns:
        distance [B] (and the accumulator if requested).
    """
    if query.ndim not in (4, 5) or support.ndim != 3:
        raise ValueError("query must be [B,K,T,D] or [B,K1,K2,T,D] and support [B,U,D]")
    one_axis = query.ndim == 4
    s1, s2 = _max_shifts(max_shift, 1 if one_axis else 2)
    out, R = _jeanie_dp_features(query.unsqueeze(2) if one_axis else query, support, gamma,
                                s1, s2, metric, backend, return_accumulator)
    if not return_accumulator:
        return out
    return out, (R[:, :, 0] if one_axis else R)


def jeanie_from_cost(cost, *, gamma=0.1, max_shift=1, return_accumulator=False,
                     backend="auto"):
    """JEANIE on a precomputed cost tensor.

    Args:
        cost: [B, K, T, U] (one viewpoint axis) or [B, K1, K2, T, U] (two
            axes), e.g. from ``euclidean_cost``.
        gamma, max_shift, return_accumulator, backend: as in ``jeanie``.

    Returns:
        distance [B] (and the accumulator if requested).
    """
    if not cost.is_floating_point():
        raise TypeError("cost must be a floating-point tensor")
    if cost.ndim not in (4, 5):
        raise ValueError("cost must be [B,K,T,U] or [B,K1,K2,T,U]")
    one_axis = cost.ndim == 4
    s1, s2 = _max_shifts(max_shift, 1 if one_axis else 2)
    out, R = _jeanie_dp(cost.unsqueeze(2) if one_axis else cost, gamma, s1, s2, backend,
                        return_accumulator)
    if not return_accumulator:
        return out
    return out, (R[:, :, 0] if one_axis else R)
