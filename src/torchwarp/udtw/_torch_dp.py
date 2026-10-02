"""Portable uDTW dynamic program in plain PyTorch.

Same recurrences as ``csrc/udtw_cuda.cu``, vectorised over the batch and
over each anti-diagonal, so the Python loop runs N+M-1 times instead of N*M.
Runs on any PyTorch device (CPU, CUDA, ROCm, MPS, XPU). Unreachable or
out-of-band cells hold R = +inf and P = 0.
"""

import torch

_INF = float("inf")


def udtw_forward(C, Q, gamma, bandwidth):
    """Return (R_final [B], P_final [B], R [B,N,M], P [B,N,M])."""
    B, N, M = C.shape
    # Row/column 0 of the padded tables is the virtual border. Rp[:,0,0]=0
    # makes the generic update give R[0,0]=C[0,0] and P[0,0]=Q[0,0].
    Rp = C.new_full((B, N + 1, M + 1), _INF)
    Pp = C.new_zeros((B, N + 1, M + 1))
    Rp[:, 0, 0] = 0
    rows = torch.arange(N, device=C.device)

    for d in range(N + M - 1):
        i = rows[max(0, d - M + 1):min(N - 1, d) + 1]
        j = d - i
        # predecessors in padded coordinates: diag, up, left
        z = torch.stack((Rp[:, i, j], Rp[:, i, j + 1], Rp[:, i + 1, j]), 1)
        z = -z / gamma
        r = C[:, i, j] - gamma * torch.logsumexp(z, 1)
        reachable = torch.isfinite(r)
        if bandwidth > 0:
            reachable = reachable & ((i - j).abs() <= bandwidth)
        w = torch.softmax(z, 1)
        pred_p = torch.stack((Pp[:, i, j], Pp[:, i, j + 1], Pp[:, i + 1, j]), 1)
        p = Q[:, i, j] + (w * pred_p).sum(1)
        Rp[:, i + 1, j + 1] = torch.where(reachable, r, _INF)
        Pp[:, i + 1, j + 1] = torch.where(reachable, p, 0.0)

    R = Rp[:, 1:, 1:]
    P = Pp[:, 1:, 1:]
    return R[:, -1, -1], P[:, -1, -1], R, P


def udtw_backward(C, Q, R, P, grad_r, grad_p, gamma):
    """Return (dL/dC, dL/dQ) for L = grad_r * R_final + grad_p * P_final."""
    B, N, M = C.shape
    # Padded at the high end so successors (i+1, j), (i, j+1) never go
    # out of range; padding has E=-inf (weight 0) and zero adjoints.
    G = C.new_zeros((B, N + 1, M + 1))   # Rbar
    H = C.new_zeros((B, N + 1, M + 1))   # Pbar
    E = C.new_full((B, N + 1, M + 1), -_INF)  # R - C
    A = C.new_zeros((B, N + 1, M + 1))   # P - Q
    G[:, N - 1, M - 1] = grad_r
    H[:, N - 1, M - 1] = grad_p
    E[:, N - 1, M - 1] = R[:, -1, -1] - C[:, -1, -1]
    A[:, N - 1, M - 1] = P[:, -1, -1] - Q[:, -1, -1]
    rows = torch.arange(N, device=C.device)

    for d in range(N + M - 3, -1, -1):
        i = rows[max(0, d - M + 1):min(N - 1, d) + 1]
        j = d - i
        rk = R[:, i, j]
        pk = P[:, i, j]
        succ = (i + 1, j), (i, j + 1), (i + 1, j + 1)
        Es = torch.stack([E[:, a, b] for a, b in succ], 1)
        Gs = torch.stack([G[:, a, b] for a, b in succ], 1)
        Hs = torch.stack([H[:, a, b] for a, b in succ], 1)
        As = torch.stack([A[:, a, b] for a, b in succ], 1)
        w = torch.exp((Es - rk.unsqueeze(1)) / gamma)
        G[:, i, j] = (w * (Gs - Hs * (pk.unsqueeze(1) - As) / gamma)).sum(1)
        H[:, i, j] = (w * Hs).sum(1)
        E[:, i, j] = torch.where(torch.isfinite(rk), rk - C[:, i, j], -_INF)
        A[:, i, j] = pk - Q[:, i, j]

    return G[:, :N, :M], H[:, :N, :M]
