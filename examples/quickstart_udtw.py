"""uDTW as a training loss: fit sequences to targets with a learned uncertainty.

  python examples/quickstart_udtw.py
"""

import torch
import torch.nn as nn

from torchwarp import UDTW

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# B pairs of sequences [B, length, features]; the targets are time-shifted sines
B, N, M, D = 8, 30, 40, 1
t = torch.linspace(0, 6.28, M, device=device)
target = torch.sin(t + torch.rand(B, 1, device=device)).unsqueeze(-1)       # [B, M, 1]

pred = nn.Parameter(torch.zeros(B, N, D, device=device))                    # [B, N, 1]
# SigmaNet: one uncertainty sigma per time step, [B, length, 1]
# sigma must be positive: Softplus plus a floor of 0.5
sigma_net = nn.Sequential(nn.Linear(D, 16), nn.ReLU(), nn.Linear(16, 1), nn.Softplus()).to(device)

udtw = UDTW(gamma=0.1, beta=0.1)
opt = torch.optim.Adam([pred, *sigma_net.parameters()], lr=0.05)

for step in range(201):
    sigma_x = sigma_net(pred) + 0.5                                         # [B, N, 1]
    sigma_y = sigma_net(target) + 0.5                                       # [B, M, 1]
    d, omega = udtw(pred, target, sigma_x, sigma_y)                         # [B], [B]
    loss = (d + omega).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print("step {:3d}  loss {:8.3f}  mean sigma {:.3f}".format(
            step, loss.item(), sigma_x.mean().item()))
