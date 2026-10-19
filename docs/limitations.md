# Limitations

torchwarp is faster than the original uDTW and JEANIE code because of how
it computes, not what it computes: the dynamic programs run along
anti-diagonals in fused CUDA kernels, and the backward passes are
analytic. The mathematics is unchanged, so in float64 torchwarp matches
the official implementations to within 1e-12 (relative), values and
gradients. Every difference
described below comes from floating-point precision, from the order of
floating-point operations, or from the size and shape limits of the
kernels.

All numbers on this page were measured on an NVIDIA TITAN RTX
(PyTorch 2.14.1, CUDA 13.0); the shared-memory limits and the
small-batch timings were also measured on an A100 80GB.

## Numerical precision

The dynamic program adds one cost per step along a path of about N + M
steps, so the accumulated value grows with the sequence length. In
float32 its absolute rounding error grows with it, and the soft-min
weights $\exp(-\Delta / \gamma)$ amplify that error by $1/\gamma$. The
distance itself stays accurate; the gradients lose accuracy as the
sequences get longer and as $\gamma$ gets smaller. This comes from
float32 itself, so any float32 implementation of these recurrences,
including the official code run in float32, is subject to it.

Gradient error in float32, as max |float32 − float64| / max |float64|
with the float64 result as reference:

| length N | uDTW, CUDA | uDTW, PyTorch backend | JEANIE, CUDA | JEANIE, PyTorch backend |
| --- | --- | --- | --- | --- |
| 50 | 3e-4 | 2e-3 | 9e-5 | 8e-4 |
| 200 | 9e-4 | 2e-2 | 2e-3 | 9e-3 |
| 500 | 5e-3 | 7e-2 | 8e-3 | 6e-2 |
| 1000 | 1e-2 | 0.6 | 3e-2 | 0.2 |
| 2000 | 0.1 | 2.3 | | |

uDTW: `method="diff"`, $\gamma = 0.1$, M = N + 20, 4 random-normal
features per step, $\sigma \sim U(0.5, 1.5)$. JEANIE: $\gamma = 0.1$, K = 5 viewpoints,
max shift 1, Euclidean cost on 6 random-normal features, U = N + 10. The
distance itself has a relative error below 2e-6 (CUDA) in every row.

What to do:

* Use float32 with the CUDA kernels by default. To check that float32 is
  enough for a task, run a short training in float32 and in float64 and
  compare the metrics.
* Use float64 when they differ, typically for long sequences with a
  small $\gamma$. float64 is much slower on consumer GPUs (TITAN RTX,
  uDTW, B = 8, N = 512: 3.2 ms in float32, 32 ms in float64) than on
  data-centre GPUs such as the A100.
* Avoid the PyTorch backend on the GPU: it is slower and, in float32,
  less accurate (table above).
* float16 and bfloat16 inputs are converted to float32 before the
  dynamic program; the outputs are float32.

## Mixed precision and TF32

Under `torch.autocast` the dynamic programs, and the fused cost
construction of `UDTW`, `torchwarp.functional.udtw` and
`torchwarp.functional.jeanie` / `JEANIE` (Euclidean and squared
Euclidean metrics), run in float32: their results are bitwise identical
with and without autocast.

The stand-alone cost functions are ordinary PyTorch code, so autocast
runs their matrix products in half precision. Relative error of the cost
under autocast (B = 4, 200 x 220 steps, 64 features):

| function | float16 autocast | bfloat16 autocast |
| --- | --- | --- |
| `pairwise_matrices(..., method="gemm")` | 6e-4 | 5e-3 |
| `pairwise_matrices(..., method="diff")` | 0 | 0 |
| `euclidean_cost` | 1e-4 | 1e-3 |

That error reaches the distance: 2e-6 (float16) to 2e-5 (bfloat16) for
`jeanie_from_cost(euclidean_cost(q, s))` and for `fvm`. To keep these
costs in float32 inside an autocast region, compute them under
`torch.autocast("cuda", enabled=False)`, or use the fused paths.

TF32 has the same effect on float32 matrix products on Ampere and newer
GPUs: with `torch.backends.cuda.matmul.allow_tf32 = True` or
`torch.set_float32_matmul_precision("high")` (or `"medium"`), the GEMMs
that build the cost (the fused uDTW and JEANIE paths included) use a
10-bit mantissa. PyTorch leaves TF32 off for matrix products by default
(precision `"highest"`); keep it off for these losses. This was not
measured here, as the TITAN RTX has no TF32.

## Determinism

The uDTW kernels and the JEANIE / soft-DTW / FVM kernels on a
precomputed cost (`jeanie_from_cost`, `soft_dtw_from_cost`,
`fvm_from_cost`) use no atomic operations: repeated runs on the same GPU
give bitwise identical values and gradients.

The fused JEANIE path (`torchwarp.functional.jeanie` and `JEANIE` with
the Euclidean or squared Euclidean metric, on CUDA) sums the gradient
with respect to the support sequence with `atomicAdd` over the K
viewpoints that share a support frame. The order of those additions
varies between runs, so the last bits of that gradient may differ; with
K = 64 the largest difference measured over 20 runs was 5e-16 (float64)
and 7e-8 (float32), relative to the largest gradient entry. The
distance and the gradient with respect to the query are bitwise
reproducible. `torch.use_deterministic_algorithms(True)` does not detect
this. For bitwise reproducible support gradients, compute the cost first
and use `jeanie_from_cost(euclidean_cost(query, support), ...)`, at the
price of materialising the cost tensor.

## Size limits of the CUDA kernels

Each sequence pair runs in one CUDA block of at most 256 threads, each
handling at most 64 cells of an anti-diagonal. This caps the number of
cells on one anti-diagonal at 16384:

| method | limit |
| --- | --- |
| uDTW | `min(N, M) <= 16384` |
| JEANIE | `K1 * K2 * min(T, U) <= 16384` (`K1 * K2` viewpoints) |
| soft-DTW, FVM | `min(T, U) <= 16384` |

Past the limit, `backend="auto"` and `backend="cuda"` raise a
`ValueError` that names the limit and the input sizes; torchwarp does
not switch to the PyTorch backend on its own. `backend="torch"` has no
such limit but is much slower.

Memory grows with the product of the lengths: uDTW keeps the
accumulators R and P and the Gram matrix (three B x N x M tensors) for
the backward pass, and JEANIE keeps a B x K1 x K2 x T x U accumulator.

## Performance

**Small batches.** One CUDA block handles one sequence pair, so a batch
much smaller than the number of streaming multiprocessors leaves most of
the GPU idle. In that regime, building the cost with PyTorch first (a
GEMM over the whole GPU) and running the plain dynamic program,

```python
cost, penalty, _ = TF.pairwise_matrices(X, Y, sigma_x, sigma_y, beta, method="gemm")
d, omega = TF.udtw_from_cost(cost, penalty, gamma=gamma)
```

can beat the fused `UDTW`. On an A100 (108 SMs) with B = 8 it was 15 to
38% faster for N = 256 to 4000 (float32 and float64; 4% at N = 5216) and
needed about twice the memory. With B = 64, `UDTW` was 1.3 to 2 times
faster and needed half the memory. On a TITAN RTX (72 SMs) the same
holds for float64, while in float32 with B = 8 `UDTW` was as fast or up
to 15% faster.

**Long sequences in the fused uDTW kernel.** The fused kernel keeps its
working rows in shared memory as long as they fit the GPU's per-block
limit, and in global memory beyond it, which is 8 to 15% slower at the
switch. With N = M, the switch is at:

| GPU | shared memory per block | float32 | float64 |
| --- | --- | --- | --- |
| TITAN RTX | 64 KiB | N = 1024 | N = 512 |
| A100 | 163 KiB | N = 2608 | N = 1304 |

**PyTorch backend.** On the GPU it is about two orders of magnitude
slower than the kernels (uDTW, B = 8, N = 512, float32: 3.2 ms vs
1.1 s per forward and backward pass).

## Falling back to the PyTorch backend

With `backend="auto"`, torchwarp uses the PyTorch backend instead of the
CUDA kernels when

* the inputs are on the CPU;
* the CUDA extension cannot be built (for example, no `nvcc` or no
  `ninja`): torchwarp then emits one `RuntimeWarning` with the reason;
* `TORCHWARP_NO_CUDA=1` is set.

To make sure the kernels run, check `torchwarp._cuda.available("udtw")`
and `torchwarp._cuda.available("jeanie")` (see
[Installation](installation.md#the-cuda-extensions)), or pass
`backend="cuda"`, which raises an error instead of falling back.
