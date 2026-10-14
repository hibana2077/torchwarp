# API reference

This document lists every public function and class in `torchwarp`,
with its signature, arguments, input and output shapes, and errors.

Shapes use these letters:

| letter | meaning |
| --- | --- |
| `B` | batch size (number of sequence pairs) |
| `N`, `M` | lengths of the two sequences compared by uDTW |
| `T`, `U` | lengths of the query and the support sequence (JEANIE, FVM, soft-DTW) |
| `K` | number of query viewpoints (`K1`, `K2` for two viewpoint axes; `Kq`, `Ks` for query and support viewpoints in full FVM) |
| `D` | feature dimension |

All functions are differentiable. `backend` is always one of `"auto"`,
`"cuda"` or `"torch"`, see [Common behaviour](index.md#common-behaviour).

**Compatibility.** Every name exported by the official `udtw` and
`jeanie` packages exists in `torchwarp` with the same positional
arguments, so `from udtw import uDTW` or `from jeanie import ...` can be
replaced by `from torchwarp import ...`. `SoftDTW` takes the arguments of
`SoftDTW` in pytorch-softdtw-cuda. The options torchwarp adds, and the
options of torchwarp's own functions, are keyword-only: they follow `*`
in the signatures below. In `jeanie_1d_*`, `jeanie_2d_*` and `fvm_*_1d`,
`fvm_*_2d`, "1d" and "2d" are the number of viewpoint axes.

Contents:

* uDTW: [`UDTW`](#udtw), [`udtw_from_features`](#udtw_from_features), [`udtw_from_cost`](#udtw_from_cost), [`pairwise_matrices`](#pairwise_matrices)
* JEANIE: [`JEANIE`](#jeanie), [`jeanie_1d_from_features`](#jeanie_1d_from_features), [`jeanie_1d_from_cost`](#jeanie_1d_from_cost), [`jeanie_2d_from_features`](#jeanie_2d_from_features), [`jeanie_2d_from_cost`](#jeanie_2d_from_cost)
* FVM: [`fvm_query_only_1d`](#fvm_query_only_1d), [`fvm_query_only_2d`](#fvm_query_only_2d), [`fvm_1d_from_cost`](#fvm_1d_from_cost), [`fvm_2d_from_cost`](#fvm_2d_from_cost), [`fvm_from_cost`](#fvm_from_cost)
* soft-DTW: [`SoftDTW`](#softdtw), [`soft_dtw`](#soft_dtw), [`softmin`](#softmin), [`softmin_weights`](#softmin_weights)
* Base distances: [`euclidean_cost`](#euclidean_cost), [`squared_euclidean_cost`](#squared_euclidean_cost), [`rbf_cost`](#rbf_cost)
* [`torchwarp.paths`](#torchwarppaths): [`Alignment`](#alignment), [`udtw`](#pathsudtw), [`soft_dtw`](#pathssoft_dtw), [`soft_dtw_per_view`](#pathssoft_dtw_per_view), [`fvm`](#pathsfvm), [`jeanie`](#pathsjeanie), [`optimal_path`](#pathsoptimal_path)
* [`torchwarp.plot`](#torchwarpplot): [`udtw_figure`](#plotudtw_figure), [`viewpoint_figure`](#plotviewpoint_figure), [`alignment`](#plotalignment), [`uncertainty`](#plotuncertainty), [`twoway`](#plottwoway), [`path_3d`](#plotpath_3d)

---

## uDTW

For time step i of X and j of Y, with uncertainties `sigma_i` and
`sigma'_j`, uDTW uses the variance
`Sigma_ij = 0.5 * (sigma_i^2 + sigma'_j^2)` (Eq. 8), the cost
`||x_i - y_j||^2 / Sigma_ij` and the penalty `beta * log(Sigma_ij)`
(Eq. 16). The distance is the soft minimum of the path costs (Eq. 2);
the penalty `Omega` is the sum of the penalty along the
paths, weighted by the same soft selection (Eq. 3).

### `UDTW`

```python
torchwarp.UDTW(use_cuda=False, gamma=1.0, normalize=False, bandwidth=None, *,
               backend="auto", method="gemm")
```

`torch.nn.Module` computing the uDTW distance and penalty of each pair
of sequences. `torchwarp.uDTW` is the same class, under the name used by
the official code.

| argument | type | default | description |
| --- | --- | --- | --- |
| `use_cuda` | bool | `False` | Ignored; kept for compatibility with the official API. The device of the inputs decides where the computation runs. |
| `gamma` | float | `1.0` | Soft minimum temperature `gamma > 0`. |
| `normalize` | bool | `False` | If `True`, return `d(X, Y) - 0.5 * (d(X, X) + d(Y, Y))` for both the distance and the penalty. |
| `bandwidth` | float or None | `None` | Sakoe-Chiba band: only cells with `abs(i - j) <= bandwidth` are used. `None` or `0` uses all cells. |
| `backend` | str | `"auto"` | `"auto"`, `"cuda"` or `"torch"`. |
| `method` | str | `"gemm"` | `"gemm"` builds the cost inside the fused kernel without storing `[B, N, M]` matrices; `"diff"` builds the matrices with `pairwise_matrices` first. Same result. |

**forward**`(X, Y, Sigma_X, Sigma_Y, beta=1.0)`

| argument | shape | description |
| --- | --- | --- |
| `X` | `[B, N, D]` | First batch of sequences. |
| `Y` | `[B, M, D]` | Second batch of sequences. |
| `Sigma_X` | `[B, N, 1]` | Uncertainty `sigma` (a standard deviation, `> 0`) of each time step of `X`. |
| `Sigma_Y` | `[B, M, 1]` | Uncertainty `sigma'` of each time step of `Y`. |
| `beta` | float | Penalty weight `beta >= 0`. |

**Returns** `UDTWOutput(distance, penalty)`, a named tuple of two `[B]`
tensors: the uDTW distance and the penalty `beta * Omega` of each pair
(X[b], Y[b]). `d, omega = udtw(...)` and `out.distance` both work.

**Raises** `ValueError` for wrong shapes, `gamma <= 0`, `beta < 0`,
`bandwidth < 0`, or a band so narrow that no path reaches the last cell;
`TypeError` for non-floating-point inputs.

```python
udtw = torchwarp.UDTW(gamma=1.0, normalize=True)
d, omega = udtw(X, Y, sx, sy, beta=1.0)
loss = (d + omega).mean()
```

### `udtw_from_features`

```python
torchwarp.udtw_from_features(X, Y, sigma_x, sigma_y, *, beta=1.0, gamma=1.0,
                             bandwidth=None, eps=1e-8, backend="auto")
```

Functional form of `UDTW` (without `normalize`). Arguments as in
`UDTW.forward` (`sigma_x`, `sigma_y` are `Sigma_X`, `Sigma_Y` there);
`eps` is the lower bound applied to `Sigma`.

**Returns** `UDTWOutput(distance [B], penalty [B])`.

### `udtw_from_cost`

```python
torchwarp.udtw_from_cost(cost, penalty, *, gamma=1.0, bandwidth=None, backend="auto")
```

Runs the uDTW dynamic program on matrices you build yourself, for
example with another base distance or another variance model (such as
the jointly generated variance of Eq. 9).

| argument | shape | description |
| --- | --- | --- |
| `cost` | `[B, N, M]` | Uncertainty-weighted cost of each cell, for example `d(x_i, y_j) / Sigma_ij`. |
| `penalty` | `[B, N, M]` | Penalty of each cell, for example `beta * log(Sigma_ij)`. |
| `gamma` | float | Soft minimum temperature, > 0. |
| `bandwidth` | float or None | Sakoe-Chiba band, as in `UDTW`. |

**Returns** `UDTWOutput(distance [B], penalty [B])`: the soft minimum of the path
costs, and the penalty summed along the paths with the same soft
selection.

```python
cost, penalty, Sigma = torchwarp.pairwise_matrices(X, Y, sx, sy, beta=1.0)
cos = torch.nn.functional.cosine_similarity(X.unsqueeze(2), Y.unsqueeze(1), dim=-1)
d, omega = torchwarp.udtw_from_cost((1 - cos) / Sigma, penalty, gamma=1.0)
```

### `pairwise_matrices`

```python
torchwarp.pairwise_matrices(X, Y, Sigma_X, Sigma_Y, beta=1.0, eps=1e-8, *,
                            method="gemm")
```

Builds the default uDTW matrices.

**Returns** `(cost, penalty, Sigma)`, each `[B, N, M]`:
`cost_ij = ||x_i - y_j||^2 / Sigma_ij`,
`penalty_ij = beta * log(Sigma_ij)`,
`Sigma_ij = max(0.5 * (sigma_i^2 + sigma'_j^2), eps)`.
`method` is `"gemm"` (`||x||^2 + ||y||^2 - 2 * x.y` with one batched
matrix product, low memory) or `"diff"` (forms `x - y` explicitly).

---

## JEANIE

JEANIE aligns a query seen from K simulated viewpoints with a support
sequence. A path moves through cells (t, u, k): t and u move as in DTW
and the viewpoint k may change by at most `iota` (`max_shift`) per step. The
path may start and end at any viewpoint. The distance is the soft
minimum of the path distances (Algorithm 1).

### `JEANIE`

```python
torchwarp.JEANIE(*, gamma=0.1, max_shift=1, metric="euclidean", backend="auto")
```

`torch.nn.Module` computing the JEANIE distance of each query and support
pair from their features.

| argument | type | default | description |
| --- | --- | --- | --- |
| `gamma` | float | `0.1` | Soft minimum temperature `gamma > 0`. |
| `max_shift` | int or (int, int) | `1` | `iota`-max shift. An int for one viewpoint axis; a pair `(azimuth, altitude)` for two axes. Values above K - 1 behave like K - 1. |
| `metric` | str | `"euclidean"` | Base distance between blocks: `"euclidean"`, `"sqeuclidean"` or `"rbf"` (see `rbf_cost`, `sigma = 0.5`). |
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

### `jeanie_1d_from_features`

```python
torchwarp.jeanie_1d_from_features(query, support, *, gamma=0.1, max_shift=1,
                                  metric="euclidean", return_accumulator=False,
                                  backend="auto")
```

JEANIE with one viewpoint axis, from features.

| argument | shape |
| --- | --- |
| `query` | `[K, T, D]` or `[B, K, T, D]` |
| `support` | `[U, D]` or `[B, U, D]` |

**Returns** a scalar (unbatched) or `[B]`; with
`return_accumulator=True`, `(distance, R)` with R `[B, K, T, U]`.

### `jeanie_1d_from_cost`

```python
torchwarp.jeanie_1d_from_cost(cost, gamma=0.1, max_shift=1,
                              return_accumulator=False, *, backend="auto")
```

JEANIE with one viewpoint axis, on a precomputed base distance tensor.

| argument | shape | description |
| --- | --- | --- |
| `cost` | `[K, T, U]` or `[B, K, T, U]` | Base distance between query block t at viewpoint k and support block u. |

**Returns** a scalar or `[B]`; with `return_accumulator=True`,
`(distance, R)` with R of the same shape as `cost`.

### `jeanie_2d_from_features`

```python
torchwarp.jeanie_2d_from_features(query, support, *, gamma=0.1, max_shift_az=1,
                                  max_shift_alt=1, metric="euclidean",
                                  return_accumulator=False, backend="auto")
```

JEANIE with two viewpoint axes (azimuth and altitude), from features.
`query` is `[K1, K2, T, D]` or `[B, K1, K2, T, D]`, `support` is
`[U, D]` or `[B, U, D]`. `max_shift_az` and `max_shift_alt` are the
`iota`-max shifts along the two axes.

**Returns** a scalar or `[B]` (and R `[B, K1, K2, T, U]` if requested).

### `jeanie_2d_from_cost`

```python
torchwarp.jeanie_2d_from_cost(cost, gamma=0.1, max_shift_az=1, max_shift_alt=1,
                              return_accumulator=False, *, backend="auto")
```

As `jeanie_2d_from_features`, on a cost `[K1, K2, T, U]` or
`[B, K1, K2, T, U]`.

---

## FVM

Free Viewpoint Matching takes, at every cell (t, u), the soft minimum of
the base distance over the viewpoints, and runs soft-DTW on the result
(JEANIE Eq. 13). It equals JEANIE with `max_shift = K - 1`.

### `fvm_query_only_1d`

```python
torchwarp.fvm_query_only_1d(cost, gamma=0.1, *, backend="auto")
```

FVM over the query viewpoints. `cost` is `[K, T, U]` or `[B, K, T, U]`
(for example `euclidean_cost(query, support)`).
**Returns** a scalar or `[B]`.

```python
D = torchwarp.euclidean_cost(query, support)
d_fvm = torchwarp.fvm_query_only_1d(D, gamma=0.1)
```

### `fvm_query_only_2d`

```python
torchwarp.fvm_query_only_2d(cost, gamma=0.1, *, backend="auto")
```

FVM over two query viewpoint axes; `cost` is `[K1, K2, T, U]` or
`[B, K1, K2, T, U]`. **Returns** a scalar or `[B]`.

### `fvm_1d_from_cost`

```python
torchwarp.fvm_1d_from_cost(cost, gamma=0.1, *, backend="auto")
```

FVM over query and support viewpoints; `cost` is `[Kq, Ks, T, U]` or
`[B, Kq, Ks, T, U]`. **Returns** a scalar or `[B]`.

### `fvm_2d_from_cost`

```python
torchwarp.fvm_2d_from_cost(cost, gamma=0.1, *, backend="auto")
```

FVM over two query and two support viewpoint axes; `cost` is
`[Kq1, Kq2, Ks1, Ks2, T, U]` with an optional leading batch dimension.
**Returns** a scalar or `[B]`.

### `fvm_from_cost`

```python
torchwarp.fvm_from_cost(cost, gamma=0.1, *, batched=False, backend="auto")
```

General form: every dimension except the last two (and the first one
when `batched=True`) is a viewpoint axis reduced by the soft minimum.

---

## soft-DTW

### `SoftDTW`

```python
torchwarp.SoftDTW(use_cuda=False, gamma=1.0, normalize=False, bandwidth=None,
                  dist_func=None, *, backend="auto")
```

`torch.nn.Module` computing soft-DTW of each pair of sequences, with the
arguments of `SoftDTW` in pytorch-softdtw-cuda.

| argument | type | default | description |
| --- | --- | --- | --- |
| `use_cuda` | bool | `False` | Ignored; the device of the inputs decides where the computation runs. |
| `gamma` | float | `1.0` | Soft minimum temperature `gamma > 0`. |
| `normalize` | bool | `False` | If `True`, return the soft-DTW divergence `d(X, Y) - 0.5 * (d(X, X) + d(Y, Y))`. |
| `bandwidth` | float or None | `None` | Sakoe-Chiba band, as in `UDTW`. |
| `dist_func` | callable or None | `None` | Function `(X, Y)` returning the cost `[B, N, M]`. `None`: squared Euclidean distance; the CUDA kernel computes it without storing `[B, N, M]`. |

**forward**`(X, Y)` with `X [B, N, D]`, `Y [B, M, D]`. **Returns**
`[B]`. A band so narrow that no path reaches the last cell raises
`ValueError` (pytorch-softdtw-cuda returns `inf`).

```python
sdtw = torchwarp.SoftDTW(gamma=0.1, normalize=True)
loss = sdtw(X, Y).mean()
```

### `soft_dtw`

```python
torchwarp.soft_dtw(cost, gamma=0.1, *, backend="auto")
```

soft-DTW of a cost `[T, U]` (returns a scalar) or `[B, T, U]` (returns
`[B]`).

### `softmin`

```python
torchwarp.softmin(x, gamma, dim=None)
```

`-gamma * log(sum(exp(-x / gamma)))` over `dim` (over all elements when `dim=None`).

### `softmin_weights`

```python
torchwarp.softmin_weights(values, gamma, dim=0)
```

The weights of the soft minimum, `softmax(-values / gamma)` over `dim`.

---

## Base distances

All three accept two layouts:

* JEANIE / FVM: `query [..., K, T, D]` and `support [..., U, D]` (same
  leading dimensions, or none for `support`) return `[..., K, T, U]`.
* Two sequences of the same rank, as for uDTW and soft-DTW:
  `x [..., N, D]` and `y [..., M, D]` return `[..., N, M]`.

For a support with viewpoints, add or remove the viewpoint axis with
`unsqueeze` / `squeeze`.

### `euclidean_cost`

```python
torchwarp.euclidean_cost(query, support)
```

`||q - s||`, computed from one matrix product. Its gradient at distance 0 is
0.

### `squared_euclidean_cost`

```python
torchwarp.squared_euclidean_cost(query, support)
```

`||q - s||^2`.

### `rbf_cost`

```python
torchwarp.rbf_cost(query, support, sigma=0.5)
```

`sum_d (2 - 2 * exp(-sigma * (q_d - s_d)^2))`. Stores the
`[..., K, T, U, D]` difference tensor, so it uses more memory than the
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
| `path_matrix` | `[B, T, U]` or `[B, K, T, U]` | `SoftMin_gamma`-selection of the path matrices: `sum_i softmax(-w / gamma)_i * Pi_i`, over all paths `Pi_i` with path distances `w_i`. Equal to the gradient of `distance` with respect to `cost`. |
| `optimal_path` | list of lists | Per pair, the cells `(t, u)` (or `(t, u, k)`) of the path with the smallest distance, from `(0, 0)` to `(T - 1, U - 1)`. |
| `cost` | `[B, T, U]` or `[B, K, T, U]` | The cost matrix the paths are computed on (for uDTW, `D / Sigma`). |
| `Sigma` | `[B, T, U]` | Variances `Sigma` (uDTW only, otherwise `None`). |
| `Omega` | `[B]` | Penalty `beta * Omega` (uDTW only, otherwise `None`). |
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

soft-DTW paths for a cost matrix `[T, U]` or `[B, T, U]`.

### `paths.soft_dtw_per_view`

```python
torchwarp.paths.soft_dtw_per_view(query, support, *, gamma=0.1, metric="euclidean",
                                  angles=None)
```

soft-DTW applied to each query viewpoint separately. `query` is
`[K, T, D]` or `[B, K, T, D]`, `support` is `[U, D]` or `[B, U, D]`.
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
                           beta=1.0, index=0, power=0.1, threshold=0.6)
```

One row of panels for pair `index`: the sDTW path matrix for each `gamma` in
`gammas` (same D, `Sigma = 1`), the uDTW path matrix for each `gamma`, and the
uncertainty `Sigma` on the uDTW path for the first `gamma`. Pixels are raised to the
power `power` so that faint paths are visible; `threshold` binarizes the
path for the uncertainty panel. **Returns** a `Figure`.

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
| `query`, `support` | `[K, T, D]` / `[U, D]`, or with a batch dimension. |
| `angles` | One label (degrees) per query viewpoint. |
| `max_shift` | An int, or a sequence of ints for one JEANIE panel each. |
| `soft_dtw_gamma`, `fvm_gamma`, `jeanie_gamma` | Per-method `gamma` (default `gamma`). |
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
                           hist=True, title=None)
```

The binarized path of a uDTW `Alignment` multiplied by `Sigma` (white = high
uncertainty), with a histogram of `Sigma` on the path when `hist=True`.

### `plot.twoway`

```python
torchwarp.plot.twoway(a, X, Y, *, index=0, ax=None, offset=None, step=1, title=None)
```

The two sequences of pair `index`, `Y` drawn `offset` below `X`, with a
line from `x_i` to `y_j` for every `step`-th cell (i, j)
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
