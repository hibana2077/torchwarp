# torchwarp documentation

torchwarp is an unofficial, faster PyTorch/CUDA implementation of
uncertainty-DTW (uDTW, ECCV 2022) and JEANIE (IJCV 2024), both proposed
by Lei Wang, Piotr Koniusz et al. It also provides soft-DTW and Free
Viewpoint Matching (FVM), and plots of the alignment paths.

| page | contents |
| --- | --- |
| [Installation](installation.md) | requirements, pip / uv, the CUDA extensions, environment variables, troubleshooting |
| [Usage](usage.md) | uDTW, JEANIE, FVM and soft-DTW step by step, custom distances, viewpoint simulation, visualization |
| [API reference](api.md) | every public function and class: arguments, shapes, return values, errors |
| [Benchmarks](benchmarks.md) | speed against the official code, the ECG5000 and NW-UCLA experiments and how to run them |

## Terms used in these pages

* **Sequence**: an ordered list of feature vectors, stored as a tensor
  `[length, features]` (with a batch: `[batch, length, features]`). A
  time series with one value per step has `features = 1`.
* **Time step**: one entry of a sequence. For skeleton sequences, a time
  step is usually a **temporal block**: a few consecutive frames encoded
  into one feature vector.
* **Base distance** d(x<sub>i</sub>, y<sub>j</sub>): the distance between
  time step i of one sequence and time step j of the other, for example
  the (squared) Euclidean distance. The matrix of all base distances is D.
* **Path**: a monotonic alignment of the two sequences from cell (0, 0)
  to cell (len_x - 1, len_y - 1) of D, moving by (1, 0), (0, 1) or (1, 1).
  The set of all paths is the **transportation plan**. The **path
  distance** w<sub>i</sub> is the sum of D over the cells of path
  Π<sub>i</sub>.
* **Soft minimum** SoftMin<sub>γ</sub>(w) = -γ log Σ<sub>i</sub>
  exp(-w<sub>i</sub>/γ). soft-DTW, uDTW and JEANIE are soft minima over
  path distances; `gamma` (γ > 0) controls how soft it is. As γ goes to 0
  it becomes the hard minimum (DTW).
* **Viewpoint** (JEANIE, FVM): the query skeleton sequence is rotated to
  K simulated camera viewpoints before it is encoded, which adds a
  viewpoint axis: `[views, length, features]`.

## Package layout

| module | contents |
| --- | --- |
| `torchwarp` | `uDTW`, `JEANIE`, the functional APIs, soft-DTW, FVM and base distances |
| `torchwarp.paths` | path matrices and optimal paths for visualization |
| `torchwarp.plot` | matplotlib figures of the paths (needs the `vis` extra) |

## Common behaviour

* **Differentiable**: every distance supports `backward()`, higher-order
  gradients (for example MAML-style meta-gradients), `torch.func`
  transforms, `vmap` and `torch.compile(fullgraph=True)`.
* **Backends**: every function takes `backend="auto"` (CUDA kernel when
  the inputs are float32/float64 CUDA tensors and the extension is built,
  otherwise PyTorch), `"cuda"` (CUDA kernel or an error) or `"torch"`
  (always the PyTorch implementation). The two backends give the same
  values and gradients.
* **dtypes**: float32 and float64 are computed as given; other floating
  types (float16, bfloat16) are computed in float32. Under autocast the
  dynamic programs run in float32.
* **Batching**: most functions take one sequence pair or a batch with one
  extra leading dimension; the API reference lists the accepted shapes.
