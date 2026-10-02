"""Batched base-distance helpers for JEANIE/FVM.

query:   [..., K, T, D]
support: [..., U, D]      (same leading batch dims as query, or none)
returns: [..., K, T, U]

Euclidean distances use ||q||^2 + ||s||^2 - 2 q.s (one matmul) instead of
materialising q - s, so memory stays O(K*T*U) rather than O(K*T*U*D).
"""

import torch


def _check(query, support):
    if query.ndim < 3 or support.ndim < 2:
        raise ValueError("query must be [..., K, T, D] and support [..., U, D]")
    if query.shape[-1] != support.shape[-1]:
        raise ValueError("query/support feature dimensions do not match")


def squared_euclidean_cost(query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Squared-Euclidean base distance."""
    _check(query, support)
    s = support.unsqueeze(-3)                       # [..., 1, U, D]
    q2 = query.pow(2).sum(-1, keepdim=True)         # [..., K, T, 1]
    s2 = s.pow(2).sum(-1).unsqueeze(-2)             # [..., 1, 1, U]
    cross = torch.matmul(query, s.transpose(-1, -2))
    return (q2 + s2 - 2.0 * cross).clamp_min(0)


def euclidean_cost(query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Euclidean base distance; the gradient at zero distance is 0."""
    sq = squared_euclidean_cost(query, support)
    positive = sq > 0
    return torch.where(positive, torch.where(positive, sq, 1.0).sqrt(), 0.0)


def rbf_cost(query: torch.Tensor, support: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """RBF-style base cost d(x,y) = sum_d [2 - 2 exp(-sigma (x_d - y_d)^2)].

    This per-dimension form cannot be written as a matmul, so it
    materialises the [..., K, T, U, D] difference tensor.
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    _check(query, support)
    diff = query.unsqueeze(-2) - support.unsqueeze(-3).unsqueeze(-3)
    return (2.0 - 2.0 * torch.exp(-sigma * diff.pow(2))).sum(dim=-1)
