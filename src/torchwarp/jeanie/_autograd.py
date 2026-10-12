"""Autograd-native JEANIE, used only for higher-order derivatives.

First-order gradients come from the analytic backward kernels. When someone
differentiates *through* that backward (create_graph=True, MAML-style meta
gradients, torch.func.grad of grad, ...), the backward op's own autograd
formula re-derives the gradient with this implementation, so every order is
exact.

Written out-of-place over anti-diagonals (T+U-1 vectorised steps, viewpoint
neighbourhoods via ``unfold``) with a large finite sentinel instead of +inf,
so no NaN appears in any derivative.
"""

import torch
import torch.nn.functional as F

BIG = 1e20


def jeanie(cost, s1, s2, gamma):
    """cost [B,K1,K2,T,U] -> (distance [B], accumulator [B,K1,K2,T,U])."""
    B, K1, K2, T, U = cost.shape
    w1, w2 = 2 * s1 + 1, 2 * s2 + 1
    rows = torch.arange(T, device=cost.device)
    big_row = cost.new_full((B, K1, K2, T), BIG)
    big_col = cost.new_full((B, K1, K2, 1), BIG)

    diagonals = []
    R1 = R2 = big_row
    for d in range(T + U - 1):
        u = d - rows
        valid = (u >= 0) & (u < U)
        c = cost[..., rows, u.clamp(0, U - 1)]            # [B,K1,K2,T]
        if d == 0:
            r = torch.where(valid, c, BIG)
        else:
            # temporal predecessors, row-indexed: (t-1,u), (t,u-1), (t-1,u-1)
            pred = torch.stack((
                torch.cat((big_col, R1[..., :-1]), -1),
                R1,
                torch.cat((big_col, R2[..., :-1]), -1),
            ), 0)                                          # [3,B,K1,K2,T]
            pred = F.pad(pred, (0, 0, s2, s2, s1, s1), value=BIG)
            pred = pred.unfold(2, w1, 1).unfold(3, w2, 1)  # [3,B,K1,K2,T,w1,w2]
            soft = -gamma * torch.logsumexp(-pred / gamma, dim=(0, 5, 6))
            r = torch.where(valid, c + soft, BIG)
        diagonals.append(r)
        R2, R1 = R1, r

    final = R1[..., T - 1].reshape(B, -1)
    out = -gamma * torch.logsumexp(-final / gamma, dim=1)
    Rd = torch.stack(diagonals, -2)                       # [B,K1,K2,D,T]
    t = torch.arange(T, device=cost.device).view(T, 1)
    u = torch.arange(U, device=cost.device).view(1, U)
    return out, Rd[..., t + u, t]


def cost_from_features(query, support, metric):
    """(Squared) Euclidean cost [B,K1,K2,T,U] from query [B,K1,K2,T,D] and
    support [B,U,D], with the GEMM form used by the fused kernel."""
    B, K1, K2, T, D = query.shape
    U = support.shape[1]
    q = query.reshape(B, K1 * K2 * T, D)
    G = torch.bmm(q, support.transpose(1, 2))
    sq = ((q * q).sum(-1, keepdim=True) + (support * support).sum(-1).unsqueeze(1)
          - 2.0 * G).clamp_min(0)
    if metric == 1:  # Euclidean, zero gradient at zero distance
        positive = sq > 0
        sq = torch.where(positive, torch.where(positive, sq, 1.0).sqrt(), 0.0)
    return sq.view(B, K1, K2, T, U)


def jeanie_features(query, support, s1, s2, gamma, metric):
    return jeanie(cost_from_features(query, support, metric), s1, s2, gamma)
