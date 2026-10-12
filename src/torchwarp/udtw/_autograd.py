"""Autograd-native uDTW, used only for higher-order derivatives.

First-order gradients come from the analytic backward kernels. When someone
differentiates *through* that backward (create_graph=True, MAML-style meta
gradients, torch.func.grad of grad, ...), the backward op's own autograd
formula re-derives the gradient with this implementation, so every order is
exact.

The program is written out-of-place over anti-diagonals (N+M-1 vectorised
steps) and uses a large finite sentinel instead of +inf, so no NaN appears
in any derivative of logsumexp/softmax.
"""

import torch

BIG = 1e20


def gram_parts(X, Y):
    """G = X Y^T and squared norms, as used by the fused kernel."""
    return torch.bmm(X, Y.transpose(1, 2)), (X * X).sum(-1), (Y * Y).sum(-1)


def matrices_from_features(X, Y, vx, vy, beta, eps):
    """Eqs. (8) and (16) with the GEMM form of ||x - y||^2."""
    G, x2, y2 = gram_parts(X, Y)
    sq = (x2.unsqueeze(2) + y2.unsqueeze(1) - 2.0 * G).clamp_min(0)
    var = (0.5 * (vx.unsqueeze(2) + vy.unsqueeze(1))).clamp_min(eps)
    return sq / var, beta * torch.log(var)


def udtw(cost, penalty, gamma, bandwidth):
    """Return (distance [B], penalty [B]) for [B,N,M] cost/penalty."""
    B, N, M = cost.shape
    rows = torch.arange(N, device=cost.device)
    big_col = cost.new_full((B, 1), BIG)
    zero_col = cost.new_zeros((B, 1))
    big_row = cost.new_full((B, N), BIG)
    zero_row = cost.new_zeros((B, N))

    R1 = R2 = big_row
    P1 = P2 = zero_row
    for d in range(N + M - 1):
        j = d - rows
        valid = (j >= 0) & (j < M)
        if bandwidth > 0:
            valid = valid & ((rows - j).abs() <= bandwidth)
        jc = j.clamp(0, M - 1)
        c = cost[:, rows, jc]
        q = penalty[:, rows, jc]
        if d == 0:
            r = torch.where(valid, c, BIG)
            p = torch.where(valid, q, 0.0)
        else:
            # predecessors, row-indexed: diag (i-1,j-1), up (i-1,j), left (i,j-1)
            r_pred = torch.stack((
                torch.cat((big_col, R2[:, :-1]), 1),
                torch.cat((big_col, R1[:, :-1]), 1),
                R1,
            ), 1)
            p_pred = torch.stack((
                torch.cat((zero_col, P2[:, :-1]), 1),
                torch.cat((zero_col, P1[:, :-1]), 1),
                P1,
            ), 1)
            z = -r_pred / gamma
            r = c - gamma * torch.logsumexp(z, 1)
            p = q + (torch.softmax(z, 1) * p_pred).sum(1)
            ok = valid & (r < 0.5 * BIG)
            r = torch.where(ok, r, BIG)
            p = torch.where(ok, p, 0.0)
        R2, R1 = R1, r
        P2, P1 = P1, p
    return R1[:, N - 1], P1[:, N - 1]


def udtw_features(X, Y, vx, vy, gamma, beta, eps, bandwidth):
    cost, pen = matrices_from_features(X, Y, vx, vy, beta, eps)
    return udtw(cost, pen, gamma, bandwidth)
