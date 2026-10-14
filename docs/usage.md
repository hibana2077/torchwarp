# Usage

Step-by-step usage of torchwarp. For the exact arguments and shapes of every function see the [API reference](api.md); for installation see [Installation](installation.md).

## uDTW

uDTW compares two sequences, for example two time series, or two
skeleton sequences encoded as one feature vector per temporal block.
Each time step i of a sequence has an uncertainty $\mathrm{sigma}_i > 0$,
predicted from the features by a small network (SigmaNet in the
paper). For time steps i of X and j of Y, uDTW uses the variance
$\mathrm{Sigma}_{ij} = \frac{1}{2} (\mathrm{sigma}_i^2 + \mathrm{sigma}'^2_j)$ and returns two
tensors of shape `[batch]`, one entry per pair (X[b], Y[b]):

* `d`: the uDTW distance (Eq. 2), the soft minimum over all warping
  paths of the sum of $\lVert x_i - y_j \rVert^2 / \mathrm{Sigma}_{ij}$ along the path;
* `omega`: the penalty $\mathrm{Omega}$ (Eq. 3), $\mathrm{beta}$ times the sum of
  $\log \mathrm{Sigma}_{ij}$ along the same soft-selected path.

With `normalize=True` both are returned as
$d(X, Y) - \frac{1}{2} \left[ d(X, X) + d(Y, Y) \right]$.

```python
import torch
import torch.nn as nn
import torchwarp

# two batches of sequences: [batch, length, features]
X = torch.randn(16, 30, 8, device="cuda", requires_grad=True)
Y = torch.randn(16, 40, 8, device="cuda")

# SigmaNet: one uncertainty sigma per time step, [batch, length, 1]
sigma_net = nn.Sequential(
    nn.Linear(8, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Softplus(),
).cuda()
# sigma must be positive: Softplus plus a floor of 0.5
sx = sigma_net(X) + 0.5
sy = sigma_net(Y) + 0.5

# d and omega: [batch], one distance and one penalty per pair
udtw = torchwarp.UDTW(gamma=1.0, normalize=True)
d, omega = udtw(X, Y, sx, sy, beta=1.0)
loss = (d + omega).mean()
loss.backward()
```

`uDTW` uses the squared Euclidean distance between time steps. To use
another distance between time steps, build the two `[batch, len_x, len_y]`
matrices of Eq. 2 and Eq. 3 yourself, $\mathrm{cost}_{ij} = d(x_i, y_j) / \mathrm{Sigma}_{ij}$
and $\mathrm{penalty}_{ij} = \mathrm{beta} \log \mathrm{Sigma}_{ij}$, and pass them to
`udtw_from_cost`. Here `pairwise_matrices` returns the default cost, the
penalty and $\mathrm{Sigma}$, and the cost is replaced by the cosine distance
$d(x_i, y_j) = 1 - \cos(x_i, y_j)$:

```python
# default matrices, each [batch, len_x, len_y]
cost, penalty, Sigma = torchwarp.pairwise_matrices(X, Y, sx, sy, beta=1.0)

# cosine distance between time steps instead of the squared Euclidean one
cos = nn.functional.cosine_similarity(X.unsqueeze(2), Y.unsqueeze(1), dim=-1)
d, omega = torchwarp.udtw_from_cost((1 - cos) / Sigma, penalty, gamma=1.0)
```

## JEANIE

JEANIE compares two 3D skeleton sequences over time and camera
viewpoint. The support sequence is used as recorded. The query is
rotated to K simulated camera viewpoints before it is encoded;
torchwarp does not do this rotation, so the query you pass already has
the viewpoint axis. The example below rotates the 3D joints of the
query about the vertical axis by each angle, splits both sequences into
temporal blocks of 8 frames, and encodes each block with a linear
layer:

```python
import math

# simulated camera viewpoints (degrees about the vertical axis)
angles = [-60, -30, 0, 30, 60]


def rotate(skeleton, degrees):
    a = math.radians(degrees)
    R = torch.tensor([[math.cos(a), 0.0, math.sin(a)],
                      [0.0, 1.0, 0.0],
                      [-math.sin(a), 0.0, math.cos(a)]])
    return skeleton @ R.T


# [frames, joints, 3] -> [blocks, size * joints * 3]
def blocks(skeleton, size=8, stride=4):
    return skeleton.unfold(0, size, stride).flatten(1)


# skeleton sequences: [frames, joints, 3]
query_skeleton = torch.randn(32, 20, 3)
support_skeleton = torch.randn(32, 20, 3)
encoder = nn.Linear(8 * 20 * 3, 64)

# query: [views, blocks, features], support: [blocks, features]
query = torch.stack(
    [encoder(blocks(rotate(query_skeleton, a))) for a in angles])
support = encoder(blocks(support_skeleton))

# dist: [batch]
jeanie = torchwarp.JEANIE(gamma=0.1, max_shift=1)
dist = jeanie(query.unsqueeze(0), support.unsqueeze(0))
```

Here a skeleton sequence is `[frames, joints, 3]`, `query` is
`[views, blocks, features]` and `support` is `[blocks, features]`;
`JEANIE` takes a batch, `[batch, views, blocks, features]` and
`[batch, blocks, features]`, and returns `dist` of shape `[batch]`, the
JEANIE distance of each query and support pair. `max_shift` is the
$\mathrm{iota}$-max shift: along the path, the viewpoint index changes by at most
$\mathrm{iota}$ from one step to the next.

The accumulator tensor R of Algorithm 1, `[batch, views, blocks_q,
blocks_s]`, is returned with:

```python
# R: [batch, views, blocks_q, blocks_s]
dist, R = jeanie(query.unsqueeze(0), support.unsqueeze(0),
                 return_accumulator=True)
```

With two viewpoint axes (azimuth and altitude) the query is
`[batch, views_az, views_alt, blocks, features]` and `max_shift` is a
pair. FVM takes the matrix of base distances between every query view
and the support:

```python
# base distances: [batch, views, blocks_q, blocks_s]
D = torchwarp.euclidean_cost(query.unsqueeze(0), support.unsqueeze(0))
d_fvm = torchwarp.fvm_query_only_1d(D, gamma=0.1)
```

### Example: classify the queries of a few-shot episode

In an N-way 1-shot episode every query is compared with every support
(one support per class); the class of the closest support is the
prediction. The negative distances are the logits for training:

```python
n_way, n_query = 5, 3
device = "cuda"
encoder = nn.Sequential(nn.Linear(8 * 20 * 3, 128), nn.ReLU(),
                        nn.Linear(128, 64)).to(device)
jeanie = torchwarp.JEANIE(gamma=0.1, max_shift=1)

# one skeleton sequence per support class, n_query queries per class
support_skeletons = torch.randn(n_way, 32, 20, 3)
query_skeletons = torch.randn(n_way * n_query, 32, 20, 3)
labels = torch.arange(n_way).repeat_interleave(n_query).to(device)

# supports: [n_way, blocks, features]
supports = torch.stack([encoder(blocks(sk).to(device)) for sk in support_skeletons])

# queries: [n_query_total, views, blocks, features]
queries = torch.stack([
    torch.stack([encoder(blocks(rotate(sk, a)).to(device)) for a in angles])
    for sk in query_skeletons])

# all query-support pairs as one batch -> distances [n_query_total, n_way]
nq, ns = len(queries), len(supports)
q = queries.unsqueeze(1).expand(-1, ns, -1, -1, -1).flatten(0, 1)
s = supports.unsqueeze(0).expand(nq, -1, -1, -1).flatten(0, 1)
dist = jeanie(q, s).view(nq, ns)

loss = nn.functional.cross_entropy(-dist, labels)
loss.backward()
pred = dist.argmin(1)
```

### Example: the effect of `max_shift`

`max_shift = 0` keeps one viewpoint for the whole path (the soft minimum
of soft-DTW over the viewpoints); larger shifts let the viewpoint move
more, and `max_shift = K - 1` lets it move freely, which equals FVM. The
distance can only decrease as `max_shift` grows:

```python
D = torchwarp.euclidean_cost(query.unsqueeze(0), support.unsqueeze(0))

for shift in range(len(angles)):
    d = torchwarp.jeanie_1d_from_cost(D, gamma=0.1, max_shift=shift)
    print("max_shift", shift, d.item())

print("FVM", torchwarp.fvm_query_only_1d(D, gamma=0.1).item())
```

### Example: which viewpoint the path uses

`torchwarp.paths.jeanie` returns the optimal path, with the viewpoint
index of every step:

```python
a = torchwarp.paths.jeanie(query, support, gamma=0.1, max_shift=1)

# cells (query block, support block, viewpoint index)
for t, u, k in a.optimal_path[0]:
    print(t, u, angles[k])
```

### Example: two viewpoint axes

Rotating the query about the vertical axis (azimuth) and about the
horizontal axis (altitude) gives a grid of viewpoints,
`[views_az, views_alt, blocks, features]`. `max_shift` is then a pair:

```python
def rotate_x(skeleton, degrees):
    a = math.radians(degrees)
    R = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, math.cos(a), -math.sin(a)],
                      [0.0, math.sin(a), math.cos(a)]])
    return skeleton @ R.T


azimuths = [-30, 0, 30]
altitudes = [-15, 0, 15]

# query_2d: [views_az, views_alt, blocks, features]
query_2d = torch.stack([
    torch.stack([encoder(blocks(rotate_x(rotate(query_skeleton, az), alt)).to(device))
                 for alt in altitudes])
    for az in azimuths])
support_2d = encoder(blocks(support_skeleton).to(device))

jeanie_2d = torchwarp.JEANIE(gamma=0.1, max_shift=(1, 1))
dist_2d = jeanie_2d(query_2d.unsqueeze(0), support_2d.unsqueeze(0))
```

### Example: another base distance

`metric` selects the distance between blocks (`"euclidean"`,
`"sqeuclidean"` or `"rbf"`). For any other distance, compute the tensor
of base distances yourself, `[batch, views, blocks_q, blocks_s]`, and
call `jeanie_1d_from_cost`, here with the cosine distance:

```python
jeanie_sq = torchwarp.JEANIE(gamma=0.1, max_shift=1, metric="sqeuclidean")
d_sq = jeanie_sq(query.unsqueeze(0), support.unsqueeze(0))

# cosine distance between every query block (per view) and support block
qn = nn.functional.normalize(query, dim=-1)
sn = nn.functional.normalize(support, dim=-1)
D_cos = 1 - qn @ sn.T
d_cos = torchwarp.jeanie_1d_from_cost(D_cos.unsqueeze(0), gamma=0.1, max_shift=1)
```

### Example: backend and precision

`backend="torch"` runs the PyTorch implementation even on the GPU (for
example to compare with the CUDA kernel), and float64 inputs are
computed in float64, which `torch.autograd.gradcheck` needs:

```python
q64 = torch.randn(1, 3, 4, 5, dtype=torch.float64, requires_grad=True)
s64 = torch.randn(1, 6, 5, dtype=torch.float64)

f = torchwarp.JEANIE(gamma=0.5, max_shift=1, backend="torch")
print(torch.autograd.gradcheck(lambda x: f(x, s64), (q64,)))
```

## soft-DTW and soft-DTW divergence

`SoftDTW` takes two batches of sequences and has the arguments of
`SoftDTW` in pytorch-softdtw-cuda, so code written for it runs after
changing the import. `normalize=True` gives the soft-DTW divergence
$\mathrm{sDTW}(x, y) - \frac{1}{2} \left[ \mathrm{sDTW}(x, x) + \mathrm{sDTW}(y, y) \right]$.

```python
# x, y: [batch, length, features]
x = torch.randn(4, 30, 8, device="cuda")
y = torch.randn(4, 40, 8, device="cuda")

d_sdtw = torchwarp.SoftDTW(gamma=0.1)(x, y)                  # [batch]
d_div = torchwarp.SoftDTW(gamma=0.1, normalize=True)(x, y)   # [batch]
```

`soft_dtw` takes a precomputed base distance matrix `[T, U]` or
`[B, T, U]`:

```python
d_sdtw = torchwarp.soft_dtw(torchwarp.squared_euclidean_cost(x, y), gamma=0.1)
```

## Visualization

Needs matplotlib: `pip install -e ".[plot]"`. A runnable version of the
code below, on toy data and without training, is
[`examples/plot_paths.py`](../examples/plot_paths.py).

`torchwarp.paths` returns an `Alignment` with the data to plot;
`torchwarp.plot` draws it. There are three ways to plot:

**1. `Alignment.plot`**: one panel per call, into a given axis.

```python
import torch
import matplotlib.pyplot as plt
from torchwarp import paths

x = torch.sin(torch.linspace(0, 6.28, 40)).view(1, -1, 1)      # [B, N, 1]
y = torch.sin(torch.linspace(0, 6.28, 50) - 0.8).view(1, -1, 1) # [B, M, 1]
sx, sy = torch.full_like(x, 0.5), torch.full_like(y, 0.5)       # sigma, [B, N, 1]

a = paths.udtw(x, y, sx, sy, gamma=0.1)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
a.plot("alignment", ax=axes[0])            # path matrix
a.plot("uncertainty", ax=axes[1])          # Sigma on the path (uDTW only)
a.plot("twoway", X=x, Y=y, ax=axes[2])     # the sequences joined along the optimal path
```

For viewpoint alignments (`paths.jeanie`, `paths.fvm`,
`paths.soft_dtw_per_view`), `a.plot("path_3d")` draws the optimal path in
3D (viewpoint, query time, support time).

**2. Figure functions**: the full multi-panel figures of the papers.

```python
from torchwarp import plot

fig = plot.udtw_figure(x, y, sx, sy, gammas=(0.01, 0.1))  # sDTW and uDTW per gamma, uncertainty
fig = plot.viewpoint_figure(query, support, angles, max_shift=(1, 2))  # soft-DTW, FVM, JEANIE
```

`query` is `[K, T, D]`, `support` is `[U, D]` and `angles` holds one
label per query viewpoint. To also draw skeletons, pass `query_poses`,
`support_poses` and `bones`.

**3. By hand**: the fields of `Alignment` are tensors and lists.

```python
plt.imshow(a.path_matrix[0], origin="lower")
i, j = zip(*a.optimal_path[0])
plt.plot(j, i, "w-")
```

| field | shape | what to plot | methods |
| --- | --- | --- | --- |
| `path_matrix` | `[B, T, U]`, or `[B, K, T, U]` with viewpoints | heatmap of the soft alignment | all |
| `optimal_path` | per pair, cells `(t, u)` or `(t, u, k)` | line over the heatmap; 3D path or viewpoint over time | all |
| `D` | as `path_matrix` | heatmap of the cost the paths are computed on | all |
| `Sigma` | `[B, T, U]` | heatmap of the uncertainty | uDTW |
| `distance`, `Omega` | `[B]` | values, e.g. in titles or legends | all (`Omega`: uDTW) |

`a.path_matrix` is $\mathrm{SoftMinSel}_{\mathrm{gamma}}(w, [\mathrm{Pi}_i])$: the path matrices
$\mathrm{Pi}_i$ of all paths, weighted by the soft-selector of Eq. 3 over their
path distances $w_i$. It is also the gradient
of the distance with respect to the cost matrix:

```python
with torch.no_grad():
    cost, penalty, Sigma = torchwarp.pairwise_matrices(x, y, sx, sy)
cost.requires_grad_(True)

d, omega = torchwarp.udtw_from_cost(cost, penalty, gamma=0.1)
d.sum().backward()

cost.grad  # equals a.path_matrix
```

### Figures from trained models

The paths depend on the features you pass in; with untrained features
the figures can be drawn but have no meaning. The two figures below come
from the uDTW model trained by `experiments/ecg5000_forecast.py` and the
JEANIE encoder trained by `experiments/nwucla_fewshot.py` (seed 42); the
commands are in [benchmarks](benchmarks.md#figures).

ECG5000, forecast of the trained uDTW model vs. ground truth for one
test series:

![uDTW paths on ECG5000](../assets/udtw_paths_ecg5000.png)

NW-UCLA, a query and a support sequence of the same action (donning),
JEANIE with 1-max and 2-max shift:

![soft-DTW, FVM and JEANIE paths on NW-UCLA](../assets/viewpoint_paths_nwucla.png)
