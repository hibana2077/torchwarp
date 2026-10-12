"""soft-DTW (Cuturi & Blondel, ICML 2017), computed by the JEANIE kernels."""

from typing import Optional

import torch

from .jeanie.functional import _batchify, _jeanie_dp


def softmin(x: torch.Tensor, gamma: float, dim: Optional[int] = None) -> torch.Tensor:
    """Differentiable soft minimum -gamma * logsumexp(-x / gamma)."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if dim is None:
        x = x.reshape(-1)
        dim = 0
    return -gamma * torch.logsumexp(-x / gamma, dim=dim)


def soft_dtw(cost: torch.Tensor, gamma: float = 0.1, backend: str = "auto") -> torch.Tensor:
    """soft-DTW for a [T, U] cost (scalar) or a [B, T, U] batch ([B])."""
    if not cost.is_floating_point():
        raise TypeError("cost must be a floating-point tensor")
    c, single = _batchify(cost, 2)
    out, _ = _jeanie_dp(c[:, None, None], gamma, 0, 0, backend)
    return out[0] if single else out
