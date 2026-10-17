"""Plot alignment paths of uDTW and JEANIE on toy data (needs matplotlib).

  python examples/plot_paths.py     # writes plot_paths_udtw.png, plot_paths_jeanie.png
"""

import math

import matplotlib.pyplot as plt
import torch

from torchwarp import paths, plot

torch.manual_seed(0)

# uDTW: y is x delayed, with a noisy segment that gets a larger sigma
t = torch.linspace(0, 2 * math.pi, 40)
x = torch.sin(t)
y = torch.sin(torch.linspace(0, 2 * math.pi, 50) - 0.8)
y[20:30] += 0.4 * torch.randn(10)
sigma_x = torch.full((40,), 0.5)
sigma_y = torch.full((50,), 0.5)
sigma_y[20:30] = 1.5
X, Y = x.view(1, -1, 1), y.view(1, -1, 1)                       # [B, N, D], [B, M, D]
a = paths.udtw(X, Y, sigma_x.view(1, -1, 1), sigma_y.view(1, -1, 1), gamma=0.1)

fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
a.plot("alignment", ax=axes[0], title="path matrix")
a.plot("uncertainty", ax=axes[1], title="Sigma on the path")
a.plot("twoway", X=X, Y=Y, ax=axes[2], step=2, title="optimal path")
# the same fields drawn by hand
axes[3].imshow(a.path_matrix[0], cmap="viridis", origin="lower")
i, j = zip(*a.optimal_path[0])
axes[3].plot(j, i, "w-", lw=1)
axes[3].set_title("a.path_matrix, a.optimal_path")
fig.tight_layout()
fig.savefig("plot_paths_udtw.png", dpi=150)

# JEANIE: the query is the support trajectory rotated by 30 degrees; the
# simulated viewpoints rotate it back by each angle, so -30 matches best
angles = [-60, -30, 0, 30, 60]


def rotate(seq, deg):                                           # about the vertical axis
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    R = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return seq @ R.T


support = torch.cumsum(0.3 * torch.randn(20, 3), dim=0)         # [U, D]
query = torch.stack([rotate(rotate(support, 30), g) for g in angles])  # [K, T, D]
query, support = query.unsqueeze(0), support.unsqueeze(0)       # [B, K, T, D], [B, U, D]

fig = plot.viewpoint_figure(query, support, angles, gamma=0.1, max_shift=1)
fig.savefig("plot_paths_jeanie.png", dpi=150)
j = paths.jeanie(query, support, gamma=0.1, max_shift=1, angles=angles)
print("JEANIE path viewpoints:", sorted({angles[k] for _, _, k in j.optimal_path[0]}))
