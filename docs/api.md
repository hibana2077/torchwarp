# API reference

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

Contents:

* uDTW: [`uDTW`](#udtw), [`udtw_from_features`](#udtw_from_features), [`udtw_from_matrices`](#udtw_from_matrices), [`pairwise_matrices`](#pairwise_matrices), [`squared_euclidean`](#squared_euclidean)
* JEANIE: [`JEANIE`](#jeanie), [`jeanie_1d_from_features`](#jeanie_1d_from_features), [`jeanie_1d_from_cost`](#jeanie_1d_from_cost), [`jeanie_2d_from_features`](#jeanie_2d_from_features), [`jeanie_2d_from_cost`](#jeanie_2d_from_cost), [`jeanie_dp_features`](#jeanie_dp_features), [`jeanie_dp`](#jeanie_dp)
* FVM: [`fvm_query_only_1d`](#fvm_query_only_1d), [`fvm_query_only_2d`](#fvm_query_only_2d), [`fvm_1d_from_cost`](#fvm_1d_from_cost), [`fvm_2d_from_cost`](#fvm_2d_from_cost), [`fvm_from_cost`](#fvm_from_cost)
* soft-DTW and helpers: [`soft_dtw`](#soft_dtw), [`softmin`](#softmin)
* Base distances: [`euclidean_cost`](#euclidean_cost), [`squared_euclidean_cost`](#squared_euclidean_cost), [`rbf_cost`](#rbf_cost)
* [`torchwarp.paths`](#torchwarppaths): [`Alignment`](#alignment), [`udtw`](#pathsudtw), [`sdtw`](#pathssdtw), [`sdtw_per_view`](#pathssdtw_per_view), [`fvm`](#pathsfvm), [`jeanie`](#pathsjeanie), [`optimal_path`](#pathsoptimal_path)
* [`torchwarp.plot`](#torchwarpplot): [`udtw_figure`](#plotudtw_figure), [`viewpoint_figure`](#plotviewpoint_figure), [`alignment`](#plotalignment), [`uncertainty`](#plotuncertainty), [`path_3d`](#plotpath_3d)

---

## uDTW

For time step i of X and j of Y, with uncertainties σ<sub>i</sub> and
σ'<sub>j</sub>, uDTW uses the variance
Σ<sub>ij</sub> = ½ (σ<sub>i</sub>² + σ'<sub>j</sub>²) (Eq. 8), the cost
‖x<sub>i</sub> - y<sub>j</sub>‖² / Σ<sub>ij</sub> and the penalty
β log Σ<sub>ij</sub> (Eq. 16). The distance is the soft minimum of the
path costs (Eq. 2); the penalty Ω is the sum of the penalty along the
paths, weighted by the same soft selection (Eq. 3).

### `uDTW`

```python
torchwarp.uDTW(use_cuda=False, gamma=1.0, normalize=False, bandwidth=None,
               backend="auto", distance="gemm")
```

`torch.nn.Module` computing the uDTW distance and penalty of each pair
of sequences.

| argument | type | default | description |
| --- | --- | --- | --- |
| `use_cuda` | bool | `False` | Ignored; kept for compatibility with the official API. The device of the inputs decides where the computation runs. |
| `gamma` | float | `1.0` | Soft minimum temperature γ, > 0. |
| `normalize` | bool | `False` | If `True`, return d(X, Y) - ½ [d(X, X) + d(Y, Y)] for both the distance and the penalty. |
| `bandwidth` | float or None | `None` | Sakoe-Chiba band: only cells with \|i - j\| ≤ `bandwidth` are used. `None` or `0` uses all cells. |
| `backend` | str | `"auto"` | `"auto"`, `"cuda"` or `"torch"`. |
| `distance` | str | `"gemm"` | `"gemm"` builds the cost inside the fused kernel without storing `[B, N, M]` matrices; `"diff"` builds the matrices with `pairwise_matrices` first. Same result. |

**forward**`(X, Y, Sigma_X, Sigma_Y, beta=1.0)`

| argument | shape | description |
| --- | --- | --- |
| `X` | `[B, N, D]` | First batch of sequences. |
| `Y` | `[B, M, D]` | Second batch of sequences. |
| `Sigma_X` | `[B, N, 1]` | Uncertainty σ (a standard deviation, > 0) of each time step of `X`. |
| `Sigma_Y` | `[B, M, 1]` | Uncertainty σ' of each time step of `Y`. |
| `beta` | float | Penalty weight β ≥ 0. |

**Returns** `(distance, penalty)`, both `[B]`: the uDTW distance and the
penalty β Ω of each pair (X[b], Y[b]).

**Raises** `ValueError` for wrong shapes, `gamma <= 0`, `beta < 0`,
`bandwidth < 0`, or a band so narrow that no path reaches the last cell;
`TypeError` for non-floating-point inputs.

```python
udtw = torchwarp.uDTW(gamma=1.0, normalize=True)
d, omega = udtw(X, Y, sx, sy, beta=1.0)
loss = (d + omega).mean()
```

### `udtw_from_features`

```python
torchwarp.udtw_from_features(X, Y, Sigma_X, Sigma_Y, beta=1.0, gamma=1.0,
                             bandwidth=None, eps=1e-8, backend="auto")
```

Functional form of `uDTW` (without `normalize`). Arguments as in
`uDTW.forward`; `eps` is the lower bound applied to Σ.

**Returns** `(distance [B], penalty [B])`.

### `udtw_from_matrices`

```python
torchwarp.udtw_from_matrices(cost, penalty, gamma=1.0, bandwidth=None,
                             backend="auto")
```

Runs the uDTW dynamic program on matrices you build yourself, for
example with another base distance or another variance model (such as
the jointly generated variance of Eq. 9).

| argument | shape | description |
| --- | --- | --- |
| `cost` | `[B, N, M]` | Uncertainty-weighted cost of each cell, for example d(x<sub>i</sub>, y<sub>j</sub>) / Σ<sub>ij</sub>. |
| `penalty` | `[B, N, M]` | Penalty of each cell, for example β log Σ<sub>ij</sub>. |
| `gamma` | float | Soft minimum temperature, > 0. |
| `bandwidth` | float or None | Sakoe-Chiba band, as in `uDTW`. |

**Returns** `(distance [B], penalty [B])`: the soft minimum of the path
costs, and the penalty summed along the paths with the same soft
selection.

```python
cost, penalty, Sigma = torchwarp.pairwise_matrices(X, Y, sx, sy, beta=1.0)
cos = torch.nn.functional.cosine_similarity(X.unsqueeze(2), Y.unsqueeze(1), dim=-1)
d, omega = torchwarp.udtw_from_matrices((1 - cos) / Sigma, penalty, gamma=1.0)
```

### `pairwise_matrices`

```python
torchwarp.pairwise_matrices(X, Y, Sigma_X, Sigma_Y, beta=1.0, eps=1e-8,
                            method="gemm")
```

Builds the default uDTW matrices.

**Returns** `(cost, penalty, Sigma)`, each `[B, N, M]`:
cost<sub>ij</sub> = ‖x<sub>i</sub> - y<sub>j</sub>‖² / Σ<sub>ij</sub>,
penalty<sub>ij</sub> = β log Σ<sub>ij</sub>,
Σ<sub>ij</sub> = max(½ (σ<sub>i</sub>² + σ'<sub>j</sub>²), `eps`).
`method` is passed to `squared_euclidean`.

### `squared_euclidean`

```python
torchwarp.squared_euclidean(X, Y, method="gemm")
```

Squared Euclidean distances `[B, N, M]` between `X [B, N, D]` and
`Y [B, M, D]`. `"gemm"` computes ‖x‖² + ‖y‖² - 2 x·y with one batched
matrix product (low memory); `"diff"` forms x - y explicitly.

---

## JEANIE

JEANIE aligns a query seen from K simulated viewpoints with a support
sequence. A path moves through cells (t, u, k): t and u move as in DTW
and the viewpoint k may change by at most ι (`max_shift`) per step. The
path may start and end at any viewpoint. The distance is the soft
minimum of the path distances (Algorithm 1).

### `JEANIE`

```python
torchwarp.JEANIE(gamma=0.1, max_shift=1, metric="euclidean", backend="auto")
```

`torch.nn.Module` computing the JEANIE distance of each query and support
pair from their features.

| argument | type | default | description |
| --- | --- | --- | --- |
| `gamma` | float | `0.1` | Soft minimum temperature γ, > 0. |
| `max_shift` | int or (int, int) | `1` | ι-max shift. An int for one viewpoint axis; a pair `(azimuth, altitude)` for two axes. Values above K - 1 behave like K - 1. |
| `metric` | str | `"euclidean"` | Base distance between blocks: `"euclidean"`, `"sqeuclidean"` or `"rbf"` (see `rbf_cost`, σ = 0.5). |
| `backend` | str | `"auto"` | `"auto"`, `"cuda"` or `"torch"`. |

**forward**`(query, support, return_accumulator=False)`

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
torchwarp.jeanie_1d_from_features(query, support, gamma=0.1, max_shift=1,
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
                              return_accumulator=False, backend="auto")
```

JEANIE with one viewpoint axis, on a precomputed base distance tensor.

| argument | shape | description |
| --- | --- | --- |
| `cost` | `[K, T, U]` or `[B, K, T, U]` | Base distance between query block t at viewpoint k and support block u. |

**Returns** a scalar or `[B]`; with `return_accumulator=True`,
`(distance, R)` with R of the same shape as `cost`.

### `jeanie_2d_from_features`

```python
torchwarp.jeanie_2d_from_features(query, support, gamma=0.1, max_shift_az=1,
                                  max_shift_alt=1, metric="euclidean",
                                  return_accumulator=False, backend="auto")
```

JEANIE with two viewpoint axes (azimuth and altitude), from features.
`query` is `[K1, K2, T, D]` or `[B, K1, K2, T, D]`, `support` is
`[U, D]` or `[B, U, D]`. `max_shift_az` and `max_shift_alt` are the
ι-max shifts along the two axes.

**Returns** a scalar or `[B]` (and R `[B, K1, K2, T, U]` if requested).

### `jeanie_2d_from_cost`

```python
torchwarp.jeanie_2d_from_cost(cost, gamma=0.1, max_shift_az=1, max_shift_alt=1,
                              return_accumulator=False, backend="auto")
```

As `jeanie_2d_from_features`, on a cost `[K1, K2, T, U]` or
`[B, K1, K2, T, U]`.

### `jeanie_dp_features`

```python
torchwarp.jeanie_dp_features(query, support, gamma=0.1, max_shift_1=1,
                             max_shift_2=1, metric="euclidean", backend="auto",
                             return_accumulator=False)
```

Batched JEANIE from features in the general two-axis layout, used by the
functions above. `query` is `[B, K1, K2, T, D]`, `support` is
`[B, U, D]` (one axis: `K2 = 1`, `max_shift_2 = 0`). For the Euclidean
metrics on CUDA the cost tensor is never stored.

**Returns** `(distance [B], R [B, K1, K2, T, U] or None)`; R is only
built when `return_accumulator=True`.

### `jeanie_dp`

```python
torchwarp.jeanie_dp(cost, gamma=0.1, max_shift_1=1, max_shift_2=1,
                    backend="auto", return_accumulator=False)
```

Batched JEANIE on a cost `[B, K1, K2, T, U]`.

**Returns** `(distance [B], R [B, K1, K2, T, U] or None)`.

---

## FVM

Free Viewpoint Matching takes, at every cell (t, u), the soft minimum of
the base distance over the viewpoints, and runs soft-DTW on the result
(JEANIE Eq. 13). It equals JEANIE with `max_shift = K - 1`.

### `fvm_query_only_1d`

```python
torchwarp.fvm_query_only_1d(cost, gamma=0.1, backend="auto")
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
torchwarp.fvm_query_only_2d(cost, gamma=0.1, backend="auto")
```

FVM over two query viewpoint axes; `cost` is `[K1, K2, T, U]` or
`[B, K1, K2, T, U]`. **Returns** a scalar or `[B]`.

### `fvm_1d_from_cost`

```python
torchwarp.fvm_1d_from_cost(cost, gamma=0.1, backend="auto")
```

FVM over query and support viewpoints; `cost` is `[Kq, Ks, T, U]` or
`[B, Kq, Ks, T, U]`. **Returns** a scalar or `[B]`.

### `fvm_2d_from_cost`

```python
torchwarp.fvm_2d_from_cost(cost, gamma=0.1, backend="auto")
```

FVM over two query and two support viewpoint axes; `cost` is
`[Kq1, Kq2, Ks1, Ks2, T, U]` with an optional leading batch dimension.
**Returns** a scalar or `[B]`.

### `fvm_from_cost`

```python
torchwarp.fvm_from_cost(cost, gamma=0.1, batched=False, backend="auto")
```

General form: every dimension except the last two (and the first one
when `batched=True`) is a viewpoint axis reduced by the soft minimum.

---

## soft-DTW and helpers

### `soft_dtw`

```python
torchwarp.soft_dtw(cost, gamma=0.1, backend="auto")
```

soft-DTW of a cost `[T, U]` (returns a scalar) or `[B, T, U]` (returns
`[B]`). The soft-DTW divergence of x and y is
`soft_dtw(c_xy) - 0.5 * (soft_dtw(c_xx) + soft_dtw(c_yy))`:

```python
def sdtw_div(x, y, gamma):
    c = lambda a, b: torchwarp.squared_euclidean(a, b)
    return torchwarp.soft_dtw(c(x, y), gamma) - 0.5 * (
        torchwarp.soft_dtw(c(x, x), gamma) + torchwarp.soft_dtw(c(y, y), gamma))
```

### `softmin`

```python
torchwarp.softmin(x, gamma, dim=None)
```

-γ log Σ exp(-x/γ) over `dim` (over all elements when `dim=None`).

---

## Base distances

All three take `query [..., K, T, D]` and `support [..., U, D]` (same
leading dimensions, or none for `support`) and return `[..., K, T, U]`.
For a support with viewpoints, or a query without, add or remove the
viewpoint axis with `unsqueeze` / `squeeze`.

### `euclidean_cost`

```python
torchwarp.euclidean_cost(query, support)
```

‖q - s‖, computed from one matrix product. Its gradient at distance 0 is
0.

### `squared_euclidean_cost`

```python
torchwarp.squared_euclidean_cost(query, support)
```

‖q - s‖².

### `rbf_cost`

```python
torchwarp.rbf_cost(query, support, sigma=0.5)
```

Σ<sub>d</sub> [2 - 2 exp(-σ (q<sub>d</sub> - s<sub>d</sub>)²)]. Stores the
`[..., K, T, U, D]` difference tensor, so it uses more memory than the
Euclidean distances.

---

## `torchwarp.paths`

Paths for visualization and analysis. These functions do not need
matplotlib.

### `Alignment`

Dataclass returned by the functions below (batched over the first
dimension).

| attribute | shape | description |
| --- | --- | --- |
| `distance` | `[B]` | The distance. |
| `path_matrix` | `[B, T, U]` or `[B, K, T, U]` | SoftMin<sub>γ</sub>-selection of the path matrices: Σ<sub>i</sub> softmax(-w/γ)<sub>i</sub> Π<sub>i</sub>, over all paths Π<sub>i</sub> with path distances w<sub>i</sub>. Equal to the gradient of `distance` with respect to `D`. |
| `optimal_path` | list of lists | Per pair, the cells `(t, u)` (or `(t, u, k)`) of the path with the smallest distance, from `(0, 0)` to `(T - 1, U - 1)`. |
| `D` | `[B, T, U]` or `[B, K, T, U]` | The matrix the paths are computed on (for uDTW, the cost D / Σ). |
| `Sigma` | `[B, T, U]` | Variances Σ (uDTW only, otherwise `None`). |
| `Omega` | `[B]` | Penalty β Ω (uDTW only, otherwise `None`). |
| `angles` | list | Viewpoint labels, if given. |

### `paths.udtw`

```python
torchwarp.paths.udtw(X, Y, sigma_x, sigma_y, gamma=0.1, beta=1.0)
```

uDTW paths for `X [B, N, D]`, `Y [B, M, D]`, `sigma_x [B, N, 1]`,
`sigma_y [B, M, 1]`. **Returns** an `Alignment` with `Sigma` and `Omega`.

### `paths.sdtw`

```python
torchwarp.paths.sdtw(D, gamma=0.1)
```

soft-DTW paths for a distance matrix `[T, U]` or `[B, T, U]`.

### `paths.sdtw_per_view`

```python
torchwarp.paths.sdtw_per_view(query, support, gamma=0.1, metric="euclidean",
                              angles=None)
```

soft-DTW applied to each query viewpoint separately. `query` is
`[K, T, D]` or `[B, K, T, D]`, `support` is `[U, D]` or `[B, U, D]`.
**Returns** a list of K `Alignment`s with cells `(t, u, k)`.

### `paths.fvm`

```python
torchwarp.paths.fvm(query, support, gamma=0.1, metric="euclidean", angles=None)
```

FVM paths (viewpoint free at every step). Shapes as in `sdtw_per_view`.

### `paths.jeanie`

```python
torchwarp.paths.jeanie(query, support, gamma=0.1, max_shift=1,
                       metric="euclidean", angles=None, start_view=None)
```

JEANIE paths with one viewpoint axis. `start_view` (an index) restricts
the paths to start at that viewpoint.

### `paths.optimal_path`

```python
torchwarp.paths.optimal_path(D, max_shift=0, start_view=None)
```

Paths with the smallest distance for `D [B, T, U]` (cells `(t, u)`) or
`D [B, K, T, U]` (cells `(t, u, k)`, viewpoint change at most
`max_shift` per step).

---

## `torchwarp.plot`

Needs matplotlib (`pip install -e ".[vis]"`). The panel functions draw
into a given matplotlib axis (or the current one) and return it; the
figure functions create and return a `matplotlib.figure.Figure`.

### `plot.udtw_figure`

```python
torchwarp.plot.udtw_figure(X, Y, sigma_x, sigma_y, gammas=(0.01, 0.1),
                           beta=1.0, index=0, power=0.1, threshold=0.6)
```

One row of panels for pair `index`: the sDTW path matrix for each γ in
`gammas` (same D, Σ = 1), the uDTW path matrix for each γ, and the
uncertainty Σ on the uDTW path for the first γ. Pixels are raised to the
power `power` so that faint paths are visible; `threshold` binarizes the
path for the uncertainty panel. **Returns** a `Figure`.

### `plot.viewpoint_figure`

```python
torchwarp.plot.viewpoint_figure(query, support, angles, gamma=0.1, max_shift=1,
                                metric="euclidean", query_poses=None,
                                support_poses=None, bones=None, index=0,
                                sdtw_gamma=None, fvm_gamma=None, jeanie_gamma=None,
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
| `sdtw_gamma`, `fvm_gamma`, `jeanie_gamma` | Per-method γ (default `gamma`). |
| `query_poses`, `support_poses`, `bones` | Optional skeletons: `[K, T, J, 2]` and `[U, J, 2]` 2D joint positions and a list of joint index pairs. |
| `pose_step` | Draw a skeleton every `pose_step` time steps. |
| `axes`, `titles` | Existing 3D axes (one per panel) and panel captions. |

**Returns** a `Figure`.

### `plot.alignment`

```python
torchwarp.plot.alignment(a, index=0, ax=None, power=0.1, title=None)
```

`a.path_matrix` of pair `index` in white on black (summed over
viewpoints if there are any).

### `plot.uncertainty`

```python
torchwarp.plot.uncertainty(a, index=0, ax=None, power=0.1, threshold=0.6,
                           hist=True, title=None)
```

The binarized path of a uDTW `Alignment` multiplied by Σ (white = high
uncertainty), with a histogram of Σ on the path when `hist=True`.

### `plot.path_3d`

```python
torchwarp.plot.path_3d(alignments, labels=None, index=0, ax=None, angles=None,
                       colors=None, query_poses=None, support_poses=None,
                       bones=None, legend_loc="center left", title=None,
                       elev=22, azim=-58, pose_step=1)
```

Optimal paths of one or more `Alignment`s on a 3D axis
(`projection="3d"`). `labels` are the legend entries, `elev` and `azim`
the view angles; the skeleton arguments are as in `viewpoint_figure`.
