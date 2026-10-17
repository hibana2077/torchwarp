# API reference

This document lists every public function and class in `torchwarp`,
with its signature, arguments, input and output shapes, and errors.

Shapes use these letters:

| letter | meaning |
| --- | --- |
| `B` | batch size (number of sequence pairs) |
| `N`, `M` | lengths of the two sequences compared by uDTW |
| `T`, `U` | lengths of the query and the support sequence (JEANIE, FVM, soft-DTW) |
| `K` | number of query viewpoints (`K1`, `K2` for two viewpoint axes; `K'` for support viewpoints in FVM) |
| `D` | feature dimension |

The API follows `torch.nn`: each method is a module in `torchwarp`
(hyperparameters in the constructor, tensors in `forward`) and a function
of the same name in `torchwarp.functional` (hyperparameters as
keyword-only arguments). The `*_from_cost` functions take a precomputed
cost tensor instead of features. Every input has a leading batch
dimension `B`. All functions are differentiable. `backend` is always one
of `"auto"`, `"cuda"` or `"torch"`, see
[Common behaviour](index.md#common-behaviour).

```python
import torchwarp
import torchwarp.functional as TF
```

| method | module | function | on a precomputed cost |
| --- | --- | --- | --- |
| uDTW | [`UDTW`](#udtw) | [`TF.udtw`](#tfudtw) | [`TF.udtw_from_cost`](#tfudtw_from_cost) |
| soft-DTW | [`SoftDTW`](#softdtw) | [`TF.soft_dtw`](#tfsoft_dtw) | [`TF.soft_dtw_from_cost`](#tfsoft_dtw_from_cost) |
| JEANIE | [`JEANIE`](#jeanie) | [`TF.jeanie`](#tfjeanie) | [`TF.jeanie_from_cost`](#tfjeanie_from_cost) |
| FVM | [`FVM`](#fvm) | [`TF.fvm`](#tffvm) | [`TF.fvm_from_cost`](#tffvm_from_cost) |

Also: [`TF.pairwise_matrices`](#tfpairwise_matrices),
[`TF.softmin`](#tfsoftmin), [`TF.softmin_weights`](#tfsoftmin_weights),
the [base distances](#base-distances) `euclidean_cost`,
`squared_euclidean_cost`, `rbf_cost` (in `torchwarp` and
`torchwarp.functional`), [`torchwarp.paths`](#torchwarppaths) and
[`torchwarp.plot`](#torchwarpplot).

---

## Modules

### `UDTW`

```python
torchwarp.UDTW(gamma=1.0, beta=1.0, normalize=False, bandwidth=None, *,
               method="gemm", backend="auto")
```

`torch.nn.Module` computing the uDTW distance and penalty of each pair
of sequences. For time step i of X and j of Y, with uncertainties
$\sigma_i$ and $\sigma_j'$, uDTW uses the variance
$\Sigma_{ij} = \frac{1}{2} (\sigma_i^2 + \sigma_j'^2)$ (Eq. 8), the cost
$\lVert x_i - y_j \rVert^2 / \Sigma_{ij}$ and the penalty $\beta \log \Sigma_{ij}$
(Eq. 16). The distance is the soft minimum of the path costs (Eq. 2);
the penalty $\Omega$ is the sum of the penalty along the
paths, weighted by the same soft selection (Eq. 3).

| argument | type | default | description |
| --- | --- | --- | --- |
| `gamma` | float | `1.0` | Soft minimum temperature $\gamma > 0$. |
| `beta` | float | `1.0` | Penalty weight $\beta \geq 0$. |
| `normalize` | bool | `False` | If `True`, return $d(X, Y) - \frac{1}{2} \left[ d(X, X) + d(Y, Y) \right]$ for both the distance and the penalty. |
| `bandwidth` | float or None | `None` | Sakoe-Chiba band: only cells with $\lvert i - j \rvert \leq$ `bandwidth` are used. `None` or `0` uses all cells. |
| `method` | str | `"gemm"` | `"gemm"` builds the cost inside the fused kernel without storing `[B, N, M]` matrices; `"diff"` builds the matrices with `pairwise_matrices` first. Same result. |
| `backend` | str | `"auto"` | `"auto"`, `"cuda"` or `"torch"`. |

**forward**`(X, Y, sigma_x, sigma_y)`

| argument | shape | description |
| --- | --- | --- |
| `X` | `[B, N, D]` | First batch of sequences. |
| `Y` | `[B, M, D]` | Second batch of sequences. |
| `sigma_x` | `[B, N, 1]` | Uncertainty $\sigma$ (a standard deviation, $> 0$) of each time step of `X`. |
| `sigma_y` | `[B, M, 1]` | Uncertainty $\sigma'$ of each time step of `Y`. |

**Returns** `UDTWOutput(distance, penalty)`, a named tuple of two `[B]`
tensors: the uDTW distance and the penalty $\beta\Omega$ of each pair
(X[b], Y[b]). `d, omega = udtw(...)` and `out.distance` both work.

**Raises** `ValueError` for wrong shapes, $\gamma \leq 0$, $\beta < 0$,
$\mathrm{bandwidth} < 0$, or a band so narrow that no path reaches the last cell;
`TypeError` for non-floating-point inputs.

```python
udtw = torchwarp.UDTW(gamma=1.0, beta=1.0, normalize=True)
d, omega = udtw(X, Y, sx, sy)
loss = (d + omega).mean()
```

### `SoftDTW`

```python
torchwarp.SoftDTW(gamma=1.0, normalize=False, bandwidth=None, dist_func=None, *,
                  backend="auto")
```

`torch.nn.Module` computing soft-DTW of each pair of sequences.

| argument | type | default | description |
| --- | --- | --- | --- |
| `gamma` | float | `1.0` | Soft minimum temperature $\gamma > 0$. |
| `normalize` | bool | `False` | If `True`, return the soft-DTW divergence $d(X, Y) - \frac{1}{2} \left[ d(X, X) + d(Y, Y) \right]$. |
| `bandwidth` | float or None | `None` | Sakoe-Chiba band, as in `UDTW`. |
| `dist_func` | callable or None | `None` | Function `(X, Y)` returning the cost `[B, N, M]`. `None`: squared Euclidean distance; the CUDA kernel computes it without storing `[B, N, M]`. |
| `backend` | str | `"auto"` | `"auto"`, `"cuda"` or `"torch"`. |

**forward**`(X, Y)` with `X [B, N, D]`, `Y [B, M, D]`. **Returns**
`[B]`. A band so narrow that no path reaches the last cell raises
`ValueError`.

```python
sdtw = torchwarp.SoftDTW(gamma=0.1, normalize=True)
loss = sdtw(X, Y).mean()
```

### `JEANIE`

```python
torchwarp.JEANIE(gamma=0.1, max_shift=1, metric="euclidean", *, backend="auto")
```

`torch.nn.Module` computing the JEANIE distance of each query and support
pair from their features. JEANIE aligns a query seen from K simulated
viewpoints with a support sequence. A path moves through cells
(t, u, k): t and u move as in DTW and the viewpoint k may change by at
most $\iota$ (`max_shift`) per step. The path may start and end at any
viewpoint. The distance is the soft minimum of the path distances
(Algorithm 1).

| argument | type | default | description |
| --- | --- | --- | --- |
| `gamma` | float | `0.1` | Soft minimum temperature $\gamma > 0$. |
| `max_shift` | int or (int, int) | `1` | $\iota$-max shift. An int applies to every viewpoint axis; a pair `(azimuth, altitude)` sets the two axes of a two-axis query separately. Values above K - 1 behave like K - 1. |
| `metric` | str | `"euclidean"` | Base distance between blocks: `"euclidean"`, `"sqeuclidean"` or `"rbf"` (see `rbf_cost`, $\sigma = 0.5$). |
| `backend` | str | `"auto"` | `"auto"`, `"cuda"` or `"torch"`. |

**forward**`(query, support, *, return_accumulator=False)`

| argument | shape | description |
| --- | --- | --- |
| `query` | `[B, K, T, D]` or `[B, K1, K2, T, D]` | Query blocks for each simulated viewpoint (one or two viewpoint axes). |
| `support` | `[B, U, D]` | Support blocks. |
| `return_accumulator` | bool | Also return the accumulator R of Algorithm 1. |

**Returns** `distance [B]`, or `(distance, R)` with R `[B, K, T, U]`
(`[B, K1, K2, T, U]` for two axes). R is differentiable.

```python
jeanie = torchwarp.JEANIE(gamma=0.1, max_shift=1)
dist = jeanie(query, support)
dist, R = jeanie(query, support, return_accumulator=True)
```

### `FVM`

```python
torchwarp.FVM(gamma=0.1, metric="euclidean", *, backend="auto")
```

`torch.nn.Module` computing Free Viewpoint Matching: at every cell
(t, u), the soft minimum of the base distance over the viewpoints, then
soft-DTW on the result (JEANIE Eq. 13). With one query viewpoint axis it
equals JEANIE with `max_shift = K - 1`. Arguments as in `JEANIE`
(without `max_shift`).

**forward**`(query, support)`

| argument | shape | description |
| --- | --- | --- |
| `query` | `[B, K, T, D]`, or more viewpoint axes `[B, K1, K2, T, D]` | Query blocks for each simulated viewpoint. |
| `support` | `[B, U, D]`, or `[B, K', U, D]` with its own viewpoints | Support blocks. All query and support viewpoint pairs are reduced. |

**Returns** `distance [B]`.

```python
fvm = torchwarp.FVM(gamma=0.1)
d_fvm = fvm(query, support)
```

---

## `torchwarp.functional`

Each function below has the same arguments as the module of the same
name, with the hyperparameters as keyword-only arguments.

### `TF.udtw`

```python
TF.udtw(X, Y, sigma_x, sigma_y, *, beta=1.0, gamma=1.0, normalize=False,
        bandwidth=None, method="gemm", eps=1e-8, backend="auto")
```

Functional form of [`UDTW`](#udtw); `eps` is the lower bound applied to
$\Sigma$. **Returns** `UDTWOutput(distance [B], penalty [B])`.

### `TF.udtw_from_cost`

```python
TF.udtw_from_cost(cost, penalty, *, gamma=1.0, bandwidth=None, backend="auto")
```

Runs the uDTW dynamic program on matrices you build yourself, for
example with another base distance or another variance model (such as
the jointly generated variance of Eq. 9).

| argument | shape | description |
| --- | --- | --- |
| `cost` | `[B, N, M]` | Uncertainty-weighted cost of each cell, for example $d(x_i, y_j) / \Sigma_{ij}$. |
| `penalty` | `[B, N, M]` | Penalty of each cell, for example $\beta \log \Sigma_{ij}$. |
| `gamma` | float | Soft minimum temperature $\gamma > 0$. |
| `bandwidth` | float or None | Sakoe-Chiba band, as in `UDTW`. |

**Returns** `UDTWOutput(distance [B], penalty [B])`: the soft minimum of the path
costs, and the penalty summed along the paths with the same soft
selection.

```python
cost, penalty, Sigma = TF.pairwise_matrices(X, Y, sx, sy, beta=1.0)
cos = torch.nn.functional.cosine_similarity(X.unsqueeze(2), Y.unsqueeze(1), dim=-1)
d, omega = TF.udtw_from_cost((1 - cos) / Sigma, penalty, gamma=1.0)
```

### `TF.pairwise_matrices`

```python
TF.pairwise_matrices(X, Y, sigma_x, sigma_y, beta=1.0, eps=1e-8, *, method="gemm")
```

Builds the default uDTW matrices.

**Returns** `(cost, penalty, Sigma)`, each `[B, N, M]`:
$\mathrm{cost} _{ij} = \lVert x _i - y _j \rVert^2 / \Sigma _{ij}$,
$\mathrm{penalty} _{ij} = \beta \log \Sigma _{ij}$,
$\Sigma _{ij} = \max\left(\frac{1}{2} (\sigma_i^2 + \sigma_j'^2), \mathrm{eps}\right)$.
`method` is `"gemm"` ($\lVert x \rVert^2 + \lVert y \rVert^2 - 2\, x \cdot y$ with one batched
matrix product, low memory) or `"diff"` (forms $x - y$ explicitly).

### `TF.soft_dtw`

```python
TF.soft_dtw(X, Y, *, gamma=1.0, normalize=False, bandwidth=None, dist_func=None,
            backend="auto")
```

Functional form of [`SoftDTW`](#softdtw). **Returns** `[B]`.

### `TF.soft_dtw_from_cost`

```python
TF.soft_dtw_from_cost(cost, *, gamma=1.0, bandwidth=None, backend="auto")
```

soft-DTW of a cost `[B, T, U]`. **Returns** `[B]`.

### `TF.jeanie`

```python
TF.jeanie(query, support, *, gamma=0.1, max_shift=1, metric="euclidean",
          return_accumulator=False, backend="auto")
```

Functional form of [`JEANIE`](#jeanie). **Returns** `distance [B]`, or
`(distance, R)` with `return_accumulator=True`.

### `TF.jeanie_from_cost`

```python
TF.jeanie_from_cost(cost, *, gamma=0.1, max_shift=1, return_accumulator=False,
                    backend="auto")
```

JEANIE on a precomputed base distance tensor.

| argument | shape | description |
| --- | --- | --- |
| `cost` | `[B, K, T, U]` or `[B, K1, K2, T, U]` | Base distance between query block t at viewpoint k and support block u, for example `euclidean_cost(query, support)`. |

**Returns** `distance [B]`; with `return_accumulator=True`,
`(distance, R)` with R of the same shape as `cost`.

### `TF.fvm`

```python
TF.fvm(query, support, *, gamma=0.1, metric="euclidean", backend="auto")
```

Functional form of [`FVM`](#fvm). **Returns** `[B]`.

### `TF.fvm_from_cost`

```python
TF.fvm_from_cost(cost, *, gamma=0.1, backend="auto")
```

FVM on a precomputed cost `[B, K..., T, U]`: every dimension between the
batch dimension and `[T, U]` is a viewpoint axis reduced by the soft
minimum, for example `[B, K, T, U]` (query viewpoints),
`[B, K1, K2, T, U]` (two query axes) or `[B, K, K', T, U]` (query and
support viewpoints). **Returns** `[B]`.

```python
D = TF.euclidean_cost(query, support)   # [B, K, T, U]
d_fvm = TF.fvm_from_cost(D, gamma=0.1)
```

### `TF.softmin`

```python
TF.softmin(x, gamma, dim=None)
```

$-\gamma \log \sum \exp(-x / \gamma)$ over `dim` (over all elements when `dim=None`).

### `TF.softmin_weights`

```python
TF.softmin_weights(values, gamma, dim=0)
```

The weights of the soft minimum, $\mathrm{softmax}(-\mathrm{values} / \gamma)$ over `dim`.

---

## Base distances

Available as `torchwarp.<name>` and `TF.<name>`. All three accept two
layouts:

* JEANIE / FVM: `query [B, K, T, D]` and `support [B, U, D]` return
  `[B, K, T, U]`.
* Two sequences of the same rank, as for uDTW and soft-DTW:
  `x [B, N, D]` and `y [B, M, D]` return `[B, N, M]`. Leading dimensions
  broadcast, so `query[:, :, None]` and `support[:, None]` with
  viewpoints give `[B, K, K', T, U]`.

### `euclidean_cost`

```python
torchwarp.euclidean_cost(query, support)
```

$\lVert q - s \rVert$, computed from one matrix product. Its gradient at distance 0 is
0.

### `squared_euclidean_cost`

```python
torchwarp.squared_euclidean_cost(query, support)
```

$\lVert q - s \rVert^2$.

### `rbf_cost`

```python
torchwarp.rbf_cost(query, support, sigma=0.5)
```

$\sum_d \left[ 2 - 2 \exp\left(-\sigma (q_d - s_d)^2\right) \right]$. Stores the
`[B, K, T, U, D]` difference tensor, so it uses more memory than the
Euclidean distances.

---

## `torchwarp.paths`

Paths for visualization and analysis, as tensors.

### `Alignment`

Dataclass returned by the functions below (batched over the first
dimension).

| attribute | shape | description |
| --- | --- | --- |
| `distance` | `[B]` | The distance. |
| `path_matrix` | `[B, T, U]` or `[B, K, T, U]` | $\mathrm{SoftMin}_{\gamma}$-selection of the path matrices: $\sum_i \mathrm{softmax}(-w / \gamma)_i \Pi_i$, over all paths $\Pi_i$ with path distances $w_i$. Equal to the gradient of `distance` with respect to `cost`. |
| `optimal_path` | list of lists | Per pair, the cells `(t, u)` (or `(t, u, k)`) of the path with the smallest distance, from `(0, 0)` to `(T - 1, U - 1)`. |
| `cost` | `[B, T, U]` or `[B, K, T, U]` | The cost matrix the paths are computed on (for uDTW, $D / \Sigma$). |
| `Sigma` | `[B, T, U]` | Variances $\Sigma$ (uDTW only, otherwise `None`). |
| `Omega` | `[B]` | Penalty $\beta\Omega$ (uDTW only, otherwise `None`). |
| `angles` | list | Viewpoint labels, if given. |

**Method** `plot(type="alignment", **kwargs)` draws the alignment with
`torchwarp.plot` (needs matplotlib) and returns the matplotlib axis.
`type` is the name of the function: `"alignment"`
([`plot.alignment`](#plotalignment)), `"uncertainty"`
([`plot.uncertainty`](#plotuncertainty), uDTW only), `"twoway"`
([`plot.twoway`](#plottwoway), pass `X` and `Y`) or `"path_3d"`
([`plot.path_3d`](#plotpath_3d)); `kwargs` go to that function.

```python
a = torchwarp.paths.udtw(X, Y, sx, sy, gamma=0.1)
a.plot("twoway", X=X, Y=Y, index=0)
```

### `paths.udtw`

```python
torchwarp.paths.udtw(X, Y, sigma_x, sigma_y, *, gamma=1.0, beta=1.0)
```

uDTW paths for `X [B, N, D]`, `Y [B, M, D]`, `sigma_x [B, N, 1]`,
`sigma_y [B, M, 1]`. **Returns** an `Alignment` with `Sigma` and `Omega`.

### `paths.soft_dtw`

```python
torchwarp.paths.soft_dtw(cost, *, gamma=0.1)
```

soft-DTW paths for a cost matrix `[B, T, U]`.

### `paths.soft_dtw_per_view`

```python
torchwarp.paths.soft_dtw_per_view(query, support, *, gamma=0.1, metric="euclidean",
                                  angles=None)
```

soft-DTW applied to each query viewpoint separately. `query` is
`[B, K, T, D]`, `support` is `[B, U, D]`.
**Returns** a list of K `Alignment`s with cells `(t, u, k)`.

### `paths.fvm`

```python
torchwarp.paths.fvm(query, support, *, gamma=0.1, metric="euclidean", angles=None)
```

FVM paths (viewpoint free at every step). Shapes as in `soft_dtw_per_view`.

### `paths.jeanie`

```python
torchwarp.paths.jeanie(query, support, *, gamma=0.1, max_shift=1,
                       metric="euclidean", angles=None, start_view=None)
```

JEANIE paths with one viewpoint axis. `start_view` (an index) restricts
the paths to start at that viewpoint.

### `paths.optimal_path`

```python
torchwarp.paths.optimal_path(cost, *, max_shift=0, start_view=None)
```

Paths with the smallest distance for `cost [B, T, U]` (cells `(t, u)`) or
`cost [B, K, T, U]` (cells `(t, u, k)`, viewpoint change at most
`max_shift` per step).

---

## `torchwarp.plot`

Needs matplotlib (`pip install torchwarp[plot]`); `import torchwarp`
does not load it. The panel functions draw
into a given matplotlib axis (or the current one) and return it; the
figure functions create and return a `matplotlib.figure.Figure`.

### `plot.udtw_figure`

```python
torchwarp.plot.udtw_figure(X, Y, sigma_x, sigma_y, *, gammas=(0.01, 0.1),
                           beta=1.0, index=0, power=0.1, threshold=0.6,
                           bin_width=0.2, std=True)
```

One row of panels for pair `index`: the sDTW path matrix for each $\gamma$ in
`gammas` (same D, $\Sigma = 1$), the uDTW path matrix for each $\gamma$, and the
uncertainty $\Sigma$ on the uDTW path for the first $\gamma$. Pixels are raised to the
power `power` so that faint paths are visible; `threshold` binarizes the
path for the uncertainty panel; `bin_width` is the width of the histogram bins and
`std` is passed to `plot.uncertainty`.
**Returns** a `Figure`.

### `plot.viewpoint_figure`

```python
torchwarp.plot.viewpoint_figure(query, support, angles, *, gamma=0.1, max_shift=1,
                                metric="euclidean", query_poses=None,
                                support_poses=None, bones=None, index=0,
                                soft_dtw_gamma=None, fvm_gamma=None, jeanie_gamma=None,
                                axes=None, titles=None, pose_step=1)
```

3D panels (viewpoint, query time, support time) for pair `index`:
soft-DTW for each viewpoint, FVM, and JEANIE with one optimal path for
each starting viewpoint.

| argument | description |
| --- | --- |
| `query`, `support` | `[B, K, T, D]` / `[B, U, D]`. |
| `angles` | One label (degrees) per query viewpoint. |
| `max_shift` | An int, or a sequence of ints for one JEANIE panel each. |
| `soft_dtw_gamma`, `fvm_gamma`, `jeanie_gamma` | Per-method $\gamma$ (default `gamma`). |
| `query_poses`, `support_poses`, `bones` | Optional skeletons: `[K, T, J, 2]` and `[U, J, 2]` 2D joint positions and a list of joint index pairs. |
| `pose_step` | Draw a skeleton every `pose_step` time steps. |
| `axes`, `titles` | Existing 3D axes (one per panel) and panel captions. |

**Returns** a `Figure`.

### `plot.alignment`

```python
torchwarp.plot.alignment(a, *, index=0, ax=None, power=0.1, title=None)
```

`a.path_matrix` of pair `index` in white on black (summed over
viewpoints if there are any).

### `plot.uncertainty`

```python
torchwarp.plot.uncertainty(a, *, index=0, ax=None, power=0.1, threshold=0.6,
                           hist=True, bin_width=0.2,
                           std=True, title=None)
```

The binarized path of a uDTW `Alignment` multiplied by $\sigma = \sqrt{\Sigma}$
(`std=True`) or by the variance $\Sigma$ (`std=False`) (white = high
uncertainty), with a histogram of these values on the path when `hist=True`
(bins and x ticks every `bin_width`, edges at multiples of `bin_width`).

### `plot.twoway`

```python
torchwarp.plot.twoway(a, X, Y, *, index=0, ax=None, offset=None, step=1, title=None)
```

The two sequences of pair `index`, `Y` drawn `offset` below `X`, with a
line from $x_i$ to $y_j$ for every `step`-th cell (i, j)
of `a.optimal_path[index]`. `X` and `Y` need one value per time step:
`[N]`, `[N, 1]` or `[B, N, 1]` (for more features, pass one, e.g.
`X[..., 0]`). `offset` defaults to 1.2 times the larger value range.

### `plot.path_3d`

```python
torchwarp.plot.path_3d(alignments, *, labels=None, index=0, ax=None, angles=None,
                       colors=None, query_poses=None, support_poses=None,
                       bones=None, legend_loc="center left", title=None,
                       elev=22, azim=-58, pose_step=1)
```

Optimal paths of one or more `Alignment`s on a 3D axis
(`projection="3d"`). `labels` are the legend entries, `elev` and `azim`
the view angles; the skeleton arguments are as in `viewpoint_figure`.
