"""JEANIE and FVM for a 5-way 1-shot episode: the nearest support is the prediction.

  python examples/quickstart_jeanie.py
"""

import torch

from torchwarp import JEANIE, euclidean_cost, fvm_query_only_1d

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Encoded temporal blocks. Each query is encoded at K simulated camera viewpoints.
n_way, n_query, K, T, D = 5, 3, 7, 8, 64
supports = torch.randn(n_way, T, D, device=device)                          # [n_way, T, D]
labels = torch.arange(n_way, device=device).repeat_interleave(n_query)      # [n_way * n_query]
# queries: noisy copies of their class support, at every viewpoint
queries = supports[labels].unsqueeze(1) + 0.5 * torch.randn(len(labels), K, T, D, device=device)

# All query-support pairs as one batch: [P, K, T, D] and [P, T, D], P = queries * n_way
q = queries.repeat_interleave(n_way, dim=0)
s = supports.repeat(len(labels), 1, 1)

jeanie = JEANIE(gamma=0.1, max_shift=1)
d_jeanie = jeanie(q, s).view(len(labels), n_way)                            # [queries, n_way]

cost = euclidean_cost(q, s)                                                 # [P, K, T, T]
d_fvm = fvm_query_only_1d(cost, gamma=0.1).view(len(labels), n_way)

for name, d in [("JEANIE", d_jeanie), ("FVM", d_fvm)]:
    acc = (d.argmin(dim=1) == labels).float().mean().item()
    print("{:6s} accuracy {:.2f}".format(name, acc))
