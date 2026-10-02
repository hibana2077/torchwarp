"""Grid <-> diagonal-major index maps used by the CUDA kernels."""

import torch

_CACHE = {}


def diag_perm(T, U, device, source="tu"):
    """Return (perm, inv) for a T x U grid of (t, u) cells.

    ``flat[:, perm]`` lists cells diagonal by diagonal (row t ascending within
    a diagonal), where ``flat`` is the grid flattened as [T, U] (``"tu"``) or
    as [U, T] (``"ut"``). ``diag[:, inv]`` restores the source order.
    """
    key = (T, U, str(device), source)
    hit = _CACHE.get(key)
    if hit is None:
        t = torch.arange(T).view(T, 1).expand(T, U).reshape(-1)
        u = torch.arange(U).view(1, U).expand(T, U).reshape(-1)
        order = torch.argsort((t + u) * T + t)
        t, u = t[order], u[order]
        perm = t * U + u if source == "tu" else u * T + t
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(T * U)
        hit = (perm.to(device), inv.to(device))
        _CACHE[key] = hit
    return hit


def diag_perm_unt(T, U, K, device):
    """(perm, inv) mapping a [U, K, T]-flattened tensor (the layout of
    ``support @ query^T`` with query in its native [K, T] order) to the
    kernel layout [T*U diagonal-major, K], and back."""
    key = (T, U, K, str(device), "unt")
    hit = _CACHE.get(key)
    if hit is None:
        cell, _ = diag_perm(T, U, "cpu", source="ut")  # u * T + t per diagonal slot
        t, u = cell % T, cell // T
        n = torch.arange(K)
        perm = (u.view(-1, 1) * K * T + n.view(1, -1) * T + t.view(-1, 1)).reshape(-1)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel())
        hit = (perm.to(device), inv.to(device))
        _CACHE[key] = hit
    return hit
