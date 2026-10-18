# Usage

Step-by-step usage of torchwarp. Like `torch.nn`, every method is a
module (`torchwarp.UDTW`, `torchwarp.SoftDTW`, `torchwarp.JEANIE`,
`torchwarp.FVM`) with a matching function in `torchwarp.functional`, and
every input has a leading batch dimension. For the exact arguments and shapes of every function see the [API reference](api.md); for installation see [Installation](installation.md). Complete training scripts and their results are in [Examples](#examples).

## uDTW

uDTW compares two sequences, for example two time series, or two
skeleton sequences encoded as one feature vector per temporal block.
It takes the two sequences and a standard deviation $\sigma_i > 0$ for
each time step i. These uncertainties can be supplied directly or
predicted by a neural network, as in the example below. For time steps
i of X and j of Y, uDTW uses the variance
$\Sigma_{ij} = \frac{1}{2} (\sigma_i^2 + \sigma_j'^2)$ and returns two
tensors of shape `[batch]`, one entry per pair (X[b], Y[b]):

* `d`: the uDTW distance, the soft minimum over all warping paths of
  the sum of $\lVert x_i - y_j \rVert^2 / \Sigma_{ij}$ along the path;
* `omega`: the penalty $\beta\Omega$, $\beta$ times the sum of
  $\log \Sigma_{ij}$ along the same soft-selected path.

With `normalize=True`, the distance and the penalty are each normalized
by subtracting their own self-alignment terms, for example
$d(X, Y) - \frac{1}{2} \left[ d(X, X) + d(Y, Y) \right]$ for the distance.

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
udtw = torchwarp.UDTW(gamma=1.0, beta=1.0, normalize=True)
d, omega = udtw(X, Y, sx, sy)
loss = (d + omega).mean()
loss.backward()
```

`UDTW` uses the squared Euclidean distance between time steps. To use
another distance between time steps, build the `[batch, N, M]` cost and
penalty matrices yourself, $\mathrm{cost} _{ij} = d(x _i, y _j) / \Sigma _{ij}$
and $\mathrm{penalty} _{ij} = \beta \log \Sigma _{ij}$, and pass them to
`TF.udtw_from_cost`. Here `TF.pairwise_matrices` returns the default cost, the
penalty and $\Sigma$, and the cost is replaced by the cosine distance
$d(x_i, y_j) = 1 - \cos(x_i, y_j)$:

```python
import torchwarp.functional as TF

# default matrices, each [batch, N, M]
cost, penalty, Sigma = TF.pairwise_matrices(X, Y, sx, sy, beta=1.0)

# cosine distance between time steps instead of the squared Euclidean one
cos = nn.functional.cosine_similarity(X.unsqueeze(2), Y.unsqueeze(1), dim=-1)
d, omega = TF.udtw_from_cost((1 - cos) / Sigma, penalty, gamma=1.0)
```

## JEANIE

JEANIE aligns two sequences while constraining how the matching
configuration of the query (for example a viewpoint or a
transformation) changes from one step to the next. This section uses
simulated camera viewpoints of 3D skeleton sequences.

The support sequence is used as recorded. The query is
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

# one pair, with a batch dimension of 1:
# query [1, views, blocks, features], support [1, blocks, features]
query = torch.stack(
    [encoder(blocks(rotate(query_skeleton, a))) for a in angles]).unsqueeze(0)
support = encoder(blocks(support_skeleton)).unsqueeze(0)

# dist: [batch]
jeanie = torchwarp.JEANIE(gamma=0.1, max_shift=1)
dist = jeanie(query, support)
```

Here a skeleton sequence is `[frames, joints, 3]`; `JEANIE` takes a
batch of queries `[batch, views, blocks, features]` and supports
`[batch, blocks, features]`, and returns `dist` of shape `[batch]`, the
JEANIE distance of each query and support pair. `max_shift` is the
largest change of the viewpoint index from one step of the path to the
next.

The dynamic programming accumulator R, `[batch, views, blocks_q,
blocks_s]`, is returned with:

```python
# R: [batch, views, blocks_q, blocks_s]
dist, R = jeanie(query, support, return_accumulator=True)
```

With two viewpoint axes (azimuth and altitude) the query is
`[batch, views_az, views_alt, blocks, features]`; an int `max_shift`
applies to both axes and a pair sets them separately. FVM lets the
viewpoint change freely at every step and takes the same inputs:

```python
fvm = torchwarp.FVM(gamma=0.1)
d_fvm = fvm(query, support)   # [batch]
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
for shift in range(len(angles)):
    d = torchwarp.JEANIE(gamma=0.1, max_shift=shift)(query, support)
    print("max_shift", shift, d.item())

print("FVM", torchwarp.FVM(gamma=0.1)(query, support).item())
```

### Example: which viewpoint the path uses

`torchwarp.paths.jeanie` returns the optimal path, with the viewpoint
index of every step:

```python
a = torchwarp.paths.jeanie(query, support, gamma=0.1, max_shift=1)

# cells (query block, support block, viewpoint index)
for i, j, k in a.optimal_path[0]:
    print(i, j, angles[k])
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

# query_2d: [1, views_az, views_alt, blocks, features]
query_2d = torch.stack([
    torch.stack([encoder(blocks(rotate_x(rotate(query_skeleton, az), alt)).to(device))
                 for alt in altitudes])
    for az in azimuths]).unsqueeze(0)
support_2d = encoder(blocks(support_skeleton).to(device)).unsqueeze(0)

jeanie_2d = torchwarp.JEANIE(gamma=0.1, max_shift=(1, 1))
dist_2d = jeanie_2d(query_2d, support_2d)
```

### Example: another base distance

`metric` selects the distance between blocks (`"euclidean"`,
`"sqeuclidean"` or `"rbf"`). For any other distance, compute the tensor
of base distances yourself, `[batch, views, blocks_q, blocks_s]`, and
call `TF.jeanie_from_cost` (or `TF.fvm_from_cost`), here with the cosine
distance:

```python
jeanie_sq = torchwarp.JEANIE(gamma=0.1, max_shift=1, metric="sqeuclidean")
d_sq = jeanie_sq(query, support)

# cosine distance between every query block (per view) and support block
qn = nn.functional.normalize(query, dim=-1)                  # [1, views, blocks_q, features]
sn = nn.functional.normalize(support, dim=-1)                # [1, blocks_s, features]
D_cos = 1 - qn @ sn.transpose(-1, -2).unsqueeze(1)           # [1, views, blocks_q, blocks_s]
d_cos = TF.jeanie_from_cost(D_cos, gamma=0.1, max_shift=1)
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

`SoftDTW` takes two batches of sequences. `normalize=True` gives the soft-DTW divergence
$\mathrm{sDTW}(x, y) - \frac{1}{2} \left[ \mathrm{sDTW}(x, x) + \mathrm{sDTW}(y, y) \right]$.

```python
# x, y: [batch, length, features]
x = torch.randn(4, 30, 8, device="cuda")
y = torch.randn(4, 40, 8, device="cuda")

d_sdtw = torchwarp.SoftDTW(gamma=0.1)(x, y)                  # [batch]
d_div = torchwarp.SoftDTW(gamma=0.1, normalize=True)(x, y)   # [batch]
```

In functional form, `TF.soft_dtw` takes the sequences and
`TF.soft_dtw_from_cost` a precomputed base distance matrix `[B, N, M]`:

```python
d_sdtw = TF.soft_dtw(x, y, gamma=0.1)
d_sdtw = TF.soft_dtw_from_cost(torchwarp.squared_euclidean_cost(x, y), gamma=0.1)
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

**2. Figure functions**: complete multi-panel figures.

```python
from torchwarp import plot

fig = plot.udtw_figure(x, y, sx, sy, gammas=(0.01, 0.1))  # sDTW and uDTW per gamma, uncertainty
fig = plot.viewpoint_figure(query, support, angles, max_shift=(1, 2))  # soft-DTW, FVM, JEANIE
```

`query` is `[B, K, N, D]`, `support` is `[B, M, D]` and `angles` holds one
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
| `path_matrix` | `[B, N, M]`, or `[B, K, N, M]` with viewpoints | heatmap of the soft alignment | all |
| `optimal_path` | per pair, cells `(i, j)` or `(i, j, k)` | line over the heatmap; 3D path or viewpoint over time | all |
| `cost` | as `path_matrix` | heatmap of the cost the paths are computed on | all |
| `Sigma` | `[B, N, M]` | heatmap of the uncertainty | uDTW |
| `distance`, `Omega` | `[B]` | values, e.g. in titles or legends | all (`Omega`: uDTW) |

`a.path_matrix` is $\sum_i \mathrm{softmax}(-w / \gamma)_i \, \Pi_i$: the path
matrices $\Pi_i$ of all paths, weighted by the soft minimum weights of
their path distances $w_i$. It is also the gradient
of the distance with respect to the cost matrix:

```python
with torch.no_grad():
    cost, penalty, Sigma = TF.pairwise_matrices(x, y, sx, sy)
cost.requires_grad_(True)

d, omega = TF.udtw_from_cost(cost, penalty, gamma=0.1)
d.sum().backward()

cost.grad  # equals a.path_matrix
```

### Figures from trained models

The paths depend on the features you pass in; with untrained features
the figures can be drawn but have no meaning. The two figures below come
from the uDTW model trained by `experiments/ecg5000_forecast.py` and the
JEANIE encoder trained by `experiments/nwucla_fewshot.py` (seed 42); the
commands are in [Figures](#figures).

ECG5000, forecast of the trained uDTW model vs. ground truth for one
test series:

![uDTW paths on ECG5000](../assets/udtw_paths_ecg5000.png)

NW-UCLA, a query and a support sequence of the same action (donning),
JEANIE with 1-max and 2-max shift:

![soft-DTW, FVM and JEANIE paths on NW-UCLA](../assets/viewpoint_paths_nwucla.png)

## Examples

Complete training scripts in [`experiments/`](../experiments/) use
torchwarp on the two tasks of the uDTW and JEANIE papers: time series
forecasting (ECG5000) and few-shot skeleton action recognition
(NW-UCLA). Their hyperparameters are the defaults at the top of each
script. Shorter scripts on toy data are in [`examples/`](../examples/):
`quickstart_udtw.py`, `quickstart_jeanie.py` and `plot_paths.py`.

### Data

```bash
bash experiments/download_data.sh
```

downloads UCR ECG5000 (about 10 MB) and the NW-UCLA Multiview skeletons
(about 14 MB) into `experiments/data/`.

### Time series forecasting (ECG5000)

UCR ECG5000 has 500 training and 4500 test series of length 140. An MLP
predicts the last 40% of each series (56 steps) from the first 60% (84
steps). Models are trained on the 500 training series and evaluated on
the 4500 test series.

Test metrics (float64, lower is better): MSE (per time step), SSE
(summed over the 56 predicted steps), DTW, soft-DTW divergence ($\gamma = 1$)
and uDTW ($\gamma = 1$, $\beta = 1$, `normalize=True`). The uDTW metric takes $\sigma$ from
one fixed SigmaNet trained with the uDTW setting (seed 100), shared by
all models.

The script has two setups. `--setup default` (the default) trains a ReLU
MLP 84 → 256 → 56 with PyTorch's default init and Adam (learning rate
1e-3, batch size 50, 100 epochs); soft-DTW and soft-DTW divergence use
$\gamma = 0.001$, uDTW $\gamma = 0.1$ and $\beta = 0.01$ without
normalisation. Results over seeds 42, 43 and 44 (mean ± std; rows are
the training losses, columns the test metrics):

```bash
python experiments/ecg5000_forecast.py --loss euclidean
python experiments/ecg5000_forecast.py --loss dtw
python experiments/ecg5000_forecast.py --loss sdtw
python experiments/ecg5000_forecast.py --loss sdtw_div
python experiments/ecg5000_forecast.py --loss udtw
```

| loss | MSE | DTW | sDTW div. | uDTW |
| --- | --- | --- | --- | --- |
| Euclidean | 0.2161 ± 0.0065 | 5.5127 ± 0.3189 | 7.8376 ± 0.3005 | 2.5342 ± 0.0911 |
| DTW | 0.7299 ± 0.1767 | 5.3317 ± 0.4190 | 19.0465 ± 3.5771 | 9.3703 ± 2.7353 |
| sDTW | 0.7265 ± 0.1751 | 5.3920 ± 0.4921 | 18.9393 ± 3.6078 | 9.2567 ± 2.6598 |
| sDTW div. | 0.7248 ± 0.1845 | 5.3684 ± 0.5537 | 19.0471 ± 3.7483 | 9.2619 ± 2.7557 |
| uDTW | 0.4003 ± 0.0742 | 5.2258 ± 0.3640 | 8.1221 ± 0.6425 | 3.2134 ± 0.3950 |

The training settings stated in the uDTW paper
([arXiv 2211.00005](https://arxiv.org/abs/2211.00005), App. H.2, H.5
and I.2) are available as `--setup paper`:

* MLP 84 → $t'$ → 56 with a tanh in between; every weight, bias included,
  drawn from N(0, 1).
* SGD, learning rate 1e-3, weight decay 1e-6.
* Per-frame SigmaNet $\sigma_t = \kappa \cdot \mathrm{sigmoid}(\mathrm{FC}(\psi_t)) + \eta$ with
  $\kappa = 1.5$ and $\eta = 0.01$ (one value per frame, as in App. H.2
  and the official `SimpleSigmaNet`), and
  $\Sigma = \frac{1}{2} (\sigma^2 + \sigma'^2)$ (Eq. 8).
* Training losses (squared Euclidean base distance): Euclidean, DTW,
  soft-DTW, soft-DTW divergence and uDTW ($d_{\mathrm{uDTW}} + \beta\Omega$, without
  normalisation).

The paper does not give $t'$, the number of epochs, the batch size, $\gamma$
or $\beta$ for this task, and tunes the remaining hyperparameters on a validation
set. [`experiments/ecg5000_tune.py --setup paper`](../experiments/ecg5000_tune.py)
does the same on a fixed 400/100 split of the training set (the test set
is not used): $t' \in \lbrace 30, 50 \rbrace$, batch size $\in \lbrace 1, 10, 50 \rbrace$,
epochs $\in \lbrace 30, 50, 100 \rbrace$, $\gamma \in \lbrace 0.01, 0.1, 1 \rbrace$,
$\beta \in \lbrace 0.01, 0.1, 1 \rbrace$, two seeds per point.
Euclidean is selected by validation MSE and the alignment losses by
validation DTW; grid points where a seed diverged are skipped. The
selected values go into `PAPER_LOSSES` in
[`experiments/ecg5000_forecast.py`](../experiments/ecg5000_forecast.py).

```bash
python experiments/ecg5000_tune.py --setup paper count            # 204 grid runs
python experiments/ecg5000_tune.py --setup paper run --task 0     # one grid run
python experiments/ecg5000_tune.py --setup paper select           # pick PAPER_LOSSES

python experiments/ecg5000_forecast.py --setup paper --loss euclidean --seeds $(seq 0 99)
python experiments/ecg5000_forecast.py --setup paper --loss dtw --seeds $(seq 0 99)
python experiments/ecg5000_forecast.py --setup paper --loss sdtw --seeds $(seq 0 99)
python experiments/ecg5000_forecast.py --setup paper --loss sdtw_div --seeds $(seq 0 99)
python experiments/ecg5000_forecast.py --setup paper --loss udtw --seeds $(seq 0 99)
```

### Few-shot action recognition (NW-UCLA)

Cross-view 5-way 1-shot action recognition on the NW-UCLA Multiview 3D
skeletons (10 actions, 20 joints, 3 Kinect views).

* Each sequence is resampled to 32 frames and split into temporal blocks
  of 8 frames with stride 4 (7 blocks). An MLP (480 → 256 → 64) encodes
  each block.
* Training: 300 episodes on classes 1 to 5, supports and queries from
  views 1 and 2, cross-entropy over the negative distances divided by a
  temperature.
* Testing: 300 episodes on classes 6, 8, 9, 11 and 12, supports from
  views 1 and 2 and five queries per class from the unseen view 3.
* soft-DTW, soft-DTW divergence and uDTW compare the 0° query with the
  support. FVM and JEANIE use the query rotated about the vertical axis
  to -60°, -30°, 0°, 30° and 60°.

```bash
python experiments/nwucla_fewshot.py --method sdtw
python experiments/nwucla_fewshot.py --method sdtw_div
python experiments/nwucla_fewshot.py --method udtw
python experiments/nwucla_fewshot.py --method fvm
python experiments/nwucla_fewshot.py --method jeanie --max-shift 1
python experiments/nwucla_fewshot.py --method jeanie --max-shift 2
```

| method | accuracy (%) |
| --- | --- |
| sDTW | 24.41 ± 2.47 |
| sDTW div. | 24.60 ± 3.56 |
| uDTW | 24.06 ± 8.43 |
| FVM | 41.15 ± 0.43 |
| JEANIE (1-max shift) | 38.75 ± 1.91 |
| JEANIE (2-max shift) | 37.08 ± 2.37 |

### Script options

| option | script | description |
| --- | --- | --- |
| `--seeds` | both | Seeds to run (default `42 43 44`). |
| `--setup` | ECG5000 | `default` or `paper` (the uDTW paper's pipeline). |
| `--val` | ECG5000 | Train on 400 training series and evaluate on the other 100. |
| `--set KEY=VALUE` | ECG5000 | Override configuration entries, e.g. `--set epochs=30 gamma=0.1`. |
| `--eval-epochs` | ECG5000 | Also evaluate after these epochs. |
| `--threads` | ECG5000 | Number of CPU threads. |
| `--device` | both | `cuda` (default when available) or `cpu`. |
| `--out` | both | Output JSON file (default `experiments/results/<task>_<method>.json`). |
| `--save-ckpt DIR` | both | Save the trained model of each seed to `DIR`. |

Each run writes the configuration, the PyTorch version, and per seed the
test metrics and the training time to the output JSON.

### Speed-up

Mean time per training step (ms) of the original implementations and of
torchwarp, on the same task, model and settings.

* **Hardware:** NVIDIA TITAN RTX; Intel Core i9-10900K, 10 CPU threads.
* **Software:** PyTorch 2.14.1 (CUDA 12.6 build), nvcc 12.4.
* **Timing:** mean over 30 training steps after 3 warm-up steps; one step
  is one 5-way 1-shot episode for NW-UCLA (a batch of 50 series for
  ECG5000), forward pass, loss, backward pass and optimizer step
  included. The GPU is synchronized before and after each step. The
  warm-up steps also trigger the one-time JIT compilation of the CUDA
  extensions, so compilation is not included.
* **Same device:** each speed-up compares the original implementation
  and torchwarp on the same device (CPU with CPU, GPU with GPU).

The original implementations run the dynamic program as Python loops
over the cells, with one small tensor operation per cell. On the GPU each
of these operations is a separate kernel launch, so launch overhead
dominates and the original code is slower on the GPU than on the CPU.
torchwarp runs the whole dynamic program in one kernel.

| method (task) | official CPU | torchwarp CPU | speed-up | official GPU | torchwarp GPU | speed-up |
| --- | --- | --- | --- | --- | --- | --- |
| uDTW (ECG5000) | 1006.9 | 88.6 | 11× | 1856.0 | 2.18 | 852× |
| uDTW (NW-UCLA) | 40.4 | 21.4 | 1.9× | 86.2 | 7.06 | 12× |
| FVM (NW-UCLA) | 163.6 | 5.53 | 30× | 386.8 | 3.75 | 103× |
| JEANIE, 1-max shift (NW-UCLA) | 886.1 | 7.40 | 120× | 2116.3 | 2.74 | 774× |
| JEANIE, 2-max shift (NW-UCLA) | 1017.6 | 11.17 | 91× | 2197.0 | 2.28 | 963× |

Where the speed comes from:

* **uDTW**: the dynamic program runs along anti-diagonals in one fused
  CUDA kernel, which also builds the cost, penalty and variance on the
  fly instead of storing `[B, N, M]` matrices; the backward pass is
  analytic rather than traced through Python loops.
* **JEANIE and FVM**: the same anti-diagonal scheme over the
  (viewpoint, query, support) volume, with the base distances computed
  inside the kernel; on the CPU, the PyTorch backend vectorises each
  anti-diagonal over the batch and the viewpoints.

The uDTW (ECG5000) row is measured with
`python experiments/ecg5000_speed.py --threads 10`.

### Figures

The two path figures of the README come from the seed 42 models:

```bash
python experiments/ecg5000_forecast.py --loss udtw --seeds 42 --save-ckpt experiments/results/ckpt
python experiments/ecg5000_plot_paths.py
python experiments/nwucla_fewshot.py --method jeanie --seeds 42 --save-ckpt experiments/results/ckpt
python experiments/nwucla_plot_paths.py --query 970 --support 1005 --out assets
```

`ecg5000_plot_paths.py` writes `assets/udtw_paths_ecg5000.png` (test series
`--index`, default 204). `nwucla_plot_paths.py` with `--query 970 --support
1005` writes `assets/viewpoint_paths_nwucla.png` (two sequences of the
action donning).

To draw the soft-DTW, FVM and JEANIE paths for the whole NW-UCLA test
split (225 queries of view 3, each with a random support of the same
action from views 1 and 2):

```bash
python experiments/nwucla_plot_paths.py
```

The figures and `index.csv` (the distances in each figure) are written to
`experiments/results/figures/nwucla/`. Options: `--support nearest` uses the
same-action support with the smallest JEANIE distance, `--max-shift`
sets the JEANIE panels (default `1 2`), `--limit N` draws only the first
N queries, and `--query I --support J` draws one pair.
