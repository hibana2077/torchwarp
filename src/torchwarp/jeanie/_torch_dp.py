"""Portable JEANIE dynamic program in plain PyTorch.

Same recurrences as ``csrc/jeanie_cuda.cu``, vectorised over the batch,
both viewpoint axes and each anti-diagonal, so the Python loop runs
T+U-1 times. The viewpoint neighbourhood is gathered with ``unfold`` on an
inf-padded accumulator. Runs on any PyTorch device.
"""

import torch

_INF = float("inf")


def jeanie_forward(C, s1, s2, gamma):
    """C: [B,K1,K2,T,U]. Returns (out [B], R [B,K1,K2,T,U])."""
    B, K1, K2, T, U = C.shape
    w1, w2 = 2 * s1 + 1, 2 * s2 + 1
    # Viewpoint axes padded by s on both sides, time axes by one leading
    # row/column; all padding is +inf ("out of range" in Algorithm 1).
    Rp = C.new_full((B, K1 + 2 * s1, K2 + 2 * s2, T + 1, U + 1), _INF)
    v1, v2 = slice(s1, s1 + K1), slice(s2, s2 + K2)
    Rp[:, v1, v2, 1, 1] = C[..., 0, 0]
    rows = torch.arange(T, device=C.device)

    for d in range(1, T + U - 1):
        t = rows[max(0, d - U + 1):min(T - 1, d) + 1]
        u = d - t
        # temporal predecessors (t-1,u), (t,u-1), (t-1,u-1) in padded coords
        pred = torch.stack((Rp[..., t, u + 1], Rp[..., t + 1, u], Rp[..., t, u]), 0)
        # [3,B,K1p,K2p,L] -> [3,B,K1,K2,L,w1,w2] viewpoint windows
        pred = pred.unfold(2, w1, 1).unfold(3, w2, 1)
        soft = -gamma * torch.logsumexp(-pred / gamma, dim=(0, 5, 6))
        Rp[:, v1, v2, t + 1, u + 1] = C[..., t, u] + soft

    R = Rp[:, v1, v2, 1:, 1:]
    final = R[..., T - 1, U - 1].reshape(B, -1)
    out = -gamma * torch.logsumexp(-final / gamma, dim=1)
    return out, R


def jeanie_backward(C, R, grad_out, s1, s2, gamma, grad_R=None):
    """Return dL/dC for L = sum_b grad_out[b] * out[b] + <grad_R, R>."""
    B, K1, K2, T, U = C.shape
    w1, w2 = 2 * s1 + 1, 2 * s2 + 1
    shape = (B, K1 + 2 * s1, K2 + 2 * s2, T + 1, U + 1)
    G = C.new_zeros(shape)              # Rbar, padding = 0
    E = C.new_full(shape, -_INF)        # R - C, padding = -inf (weight 0)
    v1, v2 = slice(s1, s1 + K1), slice(s2, s2 + K2)

    final = R[..., T - 1, U - 1]
    w_final = torch.softmax(-final.reshape(B, -1) / gamma, dim=1).reshape(B, K1, K2)
    seed = grad_out.view(B, 1, 1) * w_final
    if grad_R is not None:
        seed = seed + grad_R[..., T - 1, U - 1]
    G[:, v1, v2, T - 1, U - 1] = seed
    E[:, v1, v2, T - 1, U - 1] = final - C[..., T - 1, U - 1]
    rows = torch.arange(T, device=C.device)

    for d in range(T + U - 3, -1, -1):
        t = rows[max(0, d - U + 1):min(T - 1, d) + 1]
        u = d - t
        succ = (t + 1, u), (t, u + 1), (t + 1, u + 1)
        Es = torch.stack([E[..., a, b] for a, b in succ], 0)
        Gs = torch.stack([G[..., a, b] for a, b in succ], 0)
        Es = Es.unfold(2, w1, 1).unfold(3, w2, 1)
        Gs = Gs.unfold(2, w1, 1).unfold(3, w2, 1)
        rk = R[..., t, u]
        w = torch.exp((Es - rk[None, ..., None, None]) / gamma)
        g = (w * Gs).sum(dim=(0, 5, 6))
        if grad_R is not None:
            g = g + grad_R[..., t, u]
        G[:, v1, v2, t, u] = g
        E[:, v1, v2, t, u] = rk - C[..., t, u]

    return G[:, v1, v2, :T, :U]
