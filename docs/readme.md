# torchwarp documentation

torchwarp is an accelerated PyTorch implementation of uncertainty-DTW and JEANIE with optimized CUDA kernels.

| page | contents |
| --- | --- |
| [Installation](installation.md) | requirements, pip / uv, the CUDA extensions, environment variables, troubleshooting |
| [Usage](usage.md) | uDTW, JEANIE, FVM and soft-DTW step by step, custom distances (e.g. cosine) |
| [API reference](api.md) | Public APIs, including parameter types, defaults, valid ranges, tensor shapes, return values, and exceptions. |
| [Example](example.md) | The time-series prediction task and few-shot action recognition |

## Terms used

* **Sequence**: an ordered list of feature vectors, stored as a tensor
  `[length, features]` (with a batch: `[batch, length, features]`). A
  time series with one value per step has `features = 1`.
* **Time step**: one entry of a sequence. For skeleton sequences, a time
  step is usually a **temporal block**: a few consecutive frames encoded
  into one feature vector.
* **Base distance** $d(x_i, y_j)$: the distance between
  time step i of one sequence and time step j of the other, for example
  the (squared) Euclidean distance. The matrix of all base distances is D.
* **Path**: a monotonic alignment of the two sequences from cell (0, 0)
  to cell (len_x - 1, len_y - 1) of D, moving by (1, 0), (0, 1) or (1, 1).
  The set of all paths is the **transportation plan**. The **path
  distance** $w_i$ is the sum of D over the cells of path $\Pi_i$.
* **Soft minimum**
  $\mathrm{SoftMin}_{\gamma}(w) = -\gamma \log \sum_i \exp(-w_i / \gamma)$. soft-DTW,
  uDTW and JEANIE are soft minima over path distances; $\gamma > 0$
  controls how soft it is. As $\gamma$ goes to 0 it becomes the hard minimum (DTW).

## Package layout

| module | contents |
| --- | --- |
| `torchwarp` | the modules `UDTW`, `SoftDTW`, `JEANIE`, `FVM` and the base distances |
| `torchwarp.functional` | one function per module (`udtw`, `soft_dtw`, `jeanie`, `fvm`) and `*_from_cost` variants on a precomputed cost |
| `torchwarp.paths` | path matrices and optimal paths for visualization |
| `torchwarp.plot` | matplotlib figures of the paths, |

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
* **Batching**: every input has a leading batch dimension `B` and every
  distance is returned as `[B]`; the API reference lists the accepted
  shapes.
* **Modules and functions**: as in `torch.nn`, a module keeps its
  hyperparameters from the constructor, and the function of the same
  name in `torchwarp.functional` takes them as keyword-only arguments.
