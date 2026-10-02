"""torchwarp quickstart: uDTW with a SigmaNet, and JEANIE on skeleton blocks.

    uv run python examples/quickstart.py
"""

import torch
import torch.nn as nn

import torchwarp

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# --------------------------------------------------------------------- uDTW
# Sequences X [B,N,D], Y [B,M,D]; a SigmaNet predicts per-frame sigma > 0.
B, N, M, D = 16, 30, 40, 32
X = torch.randn(B, N, D, device=device)
Y = torch.randn(B, M, D, device=device)
sigma_net = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus()).to(device)

udtw = torchwarp.uDTW(gamma=0.1)
opt = torch.optim.Adam(sigma_net.parameters(), lr=1e-2)
for step in range(5):
    distance, penalty = udtw(X, Y, sigma_net(X) + 0.1, sigma_net(Y) + 0.1, beta=1.0)
    loss = (distance + penalty).mean()          # d_uDTW + beta * Omega
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("uDTW   step {} | loss {:.4f}".format(step, loss.item()))

# ------------------------------------------------------------------- JEANIE
# Query: K simulated viewpoints x T temporal blocks; support: U blocks.
B, K, T, U, D = 8, 5, 10, 12, 64
query = torch.randn(B, K, T, D, device=device, requires_grad=True)
support = torch.randn(B, U, D, device=device)

jeanie = torchwarp.JEANIE(gamma=0.1, max_shift=1)    # iota-max shift = 1
d, R = jeanie(query, support, return_accumulator=True)
d.sum().backward()
print("JEANIE distances", tuple(d.shape), "accumulator", tuple(R.shape),
      "| grad finite:", bool(torch.isfinite(query.grad).all()))

# Two viewpoint axes (azimuth x altitude) and soft-DTW / FVM baselines.
q2 = torch.randn(B, 3, 3, T, D, device=device)
print("JEANIE-2D", tuple(torchwarp.JEANIE(0.1, (1, 1))(q2, support).shape))
cost = torchwarp.euclidean_cost(query.detach(), support)       # [B,K,T,U]
print("FVM     ", tuple(torchwarp.fvm_query_only_1d(cost, 0.1).shape))
print("soft-DTW", tuple(torchwarp.soft_dtw(cost[:, 0], 0.1).shape))
