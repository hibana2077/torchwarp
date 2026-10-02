# torchwarp

**Fast, differentiable time & viewpoint warping for PyTorch — uDTW and JEANIE with CUDA kernels.**

[![tests](https://github.com/hibana2077/torchwarp/actions/workflows/tests.yml/badge.svg)](https://github.com/hibana2077/torchwarp/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c)

torchwarp warps, aligns and matches sequences. It is a drop-in, GPU-accelerated
implementation of two soft-DTW-family distances by Lei Wang, Piotr Koniusz et al.:

| method | what it aligns | paper |
| --- | --- | --- |
| **uDTW** — uncertainty-DTW | time, with per-frame uncertainty Σ | Wang & Koniusz, *ECCV 2022* |
| **JEANIE** | time **and** camera viewpoint, jointly | Wang et al., *IJCV 2024* |

The package also includes soft-DTW and Free Viewpoint Matching (FVM).

- **Fast:** 30–700× faster than the reference uDTW and up to 19,000× faster
  than the reference JEANIE ([numbers](#performance)).
- **Exact:** it matches the reference implementations to 1e-10 on values and
  gradients. On real-data training runs in float64, the two agree to 1e-11.
- **Complete:**
  - gradients of any order (MAML / `create_graph`)
  - forward mode
  - every `torch.func` transform (`grad`, `jacrev`, `jacfwd`, `hessian`, `vmap`)
  - `torch.compile(fullgraph=True)`
  - autocast
  - a differentiable JEANIE accumulator
- **Portable:** runs on CUDA through JIT-compiled kernels and on ROCm through
  hipify (untested on AMD hardware). On CPU, MPS or any other device it uses
  a vectorised PyTorch backend.

## Install

```bash
git clone https://github.com/hibana2077/torchwarp && cd torchwarp
uv sync --extra cu126            # or --extra cu128 / --extra cpu
```

With pip: `pip install git+https://github.com/hibana2077/torchwarp`.

The CUDA kernels are compiled on first use, which takes about 1 minute and is
then cached. This needs `nvcc` with the same major CUDA version as your torch
build. Without a working `nvcc`, torchwarp warns once and falls back to the
PyTorch backend.

## Quickstart

```python
import torch, torchwarp

# uDTW: sequences [B,N,D] / [B,M,D], per-frame sigma [B,N,1] / [B,M,1] (e.g. from a SigmaNet)
udtw = torchwarp.uDTW(gamma=0.1)
distance, penalty = udtw(X, Y, sigma_x, sigma_y, beta=1.0)
loss = (distance + penalty).mean()              # d_uDTW + beta * Omega

# JEANIE: query [B,K,T,D] (K simulated viewpoints), support [B,U,D]
jeanie = torchwarp.JEANIE(gamma=0.1, max_shift=1)   # max_shift=(1,1) for [B,K1,K2,T,D]
d = jeanie(query, support)                          # [B]
d, R = jeanie(query, support, return_accumulator=True)
```

A runnable version is in [`examples/quickstart.py`](examples/quickstart.py).

## API

| function | input → output |
| --- | --- |
| `uDTW(gamma, normalize, bandwidth)` | `(X, Y, Σx, Σy, beta)` → `(distance, penalty)`, both `[B]`. Drop-in for the reference module. |
| `udtw_from_features` / `udtw_from_matrices` | Fused path from features / DP on custom cost and penalty matrices (e.g. Eq. 9). |
| `JEANIE(gamma, max_shift, metric)` | `(query, support)` → `[B]` |
| `jeanie_{1d,2d}_from_features` | Same as `JEANIE`, from features. The cost tensor is never materialised. |
| `jeanie_{1d,2d}_from_cost`, `jeanie_dp` | DP on a cost tensor `[(B,) K(,K2), T, U]` |
| `soft_dtw`, `fvm_*` | soft-DTW `[(B,) T, U]`; Free Viewpoint Matching (Eq. 13) |
| `euclidean_cost`, `squared_euclidean_cost`, `rbf_cost` | Base distances `[..., K, T, D] × [..., U, D]` |

Every function accepts an optional leading batch dimension, and
`backend="auto" | "cuda" | "torch"`. Set `TORCHWARP_FAST_MATH=1` for faster
float32 `exp`/`log` (10–18% on the DP).

## Performance

Measured on a TITAN RTX with float32, forward + backward. The reference runs on CPU, its fastest device.

| case | reference | torchwarp (CUDA) | speed-up |
| --- | --- | --- | --- |
| uDTW, B=8, N=M=50 | 626 ms | 0.89 ms | **703×** |
| uDTW, B=8, N=M=25, normalize | 438 ms | 2.40 ms | **182×** |
| JEANIE-1D, B=8, K=5, T=U=20 | 2,346 ms | 0.68 ms | **3,472×** |
| JEANIE-2D, B=8, 5×5 views, T=U=20 | 15,675 ms | 0.82 ms | **19,061×** |
| uDTW training, ECG5000, 1000 steps | 878 s | 2.4 s | **366×** |
| JEANIE few-shot training, NW-UCLA, 300 episodes | 275 s | 0.66 s | **416×** |

At small sizes, run time is dominated by Python overhead (about 0.8 ms per
call). `torch.compile(..., mode="reduce-overhead")` brings this down to about
0.4 ms. To reproduce the table, run `benchmarks/speed_{udtw,jeanie}.py`.

## Correctness

- **Unit tests.** `pytest` (459 tests) compares every function, backend,
  gradient, higher-order derivative and `torch.func` / `torch.compile` path
  against the original code from [LeiWangR/uDTW](https://github.com/LeiWangR/uDTW)
  and [LeiWangR/JEANIE](https://github.com/LeiWangR/JEANIE). That code is
  downloaded at a pinned commit; it is not redistributed with torchwarp.
- **Real-data training.** Reference and torchwarp were trained with the same
  seeds, data and hyperparameters on two tasks. Full details are in
  [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).
  - **uDTW**, ECG5000 forecasting (as in the uDTW paper): test MSE is 0.4271
    for the reference and 0.4270 for torchwarp.
  - **JEANIE**, NW-UCLA cross-view 5-way 1-shot: accuracy is 35.3% for the
    reference and 35.6% for torchwarp.
  - **float64:** the two implementations agree to ≤1e-11.

## How it works

- **One thread block per sequence pair** sweeps anti-diagonals. Only three
  diagonals are kept in shared memory, which keeps occupancy high.
- **Diagonal-major memory layout** (viewpoints innermost for JEANIE) with
  prefetch makes every global access coalesced.
- **Fused cost construction.** One GEMM computes ⟨x, y⟩; the uncertainty
  weighting (`‖x−y‖²/Σ`, `β log Σ`) or the Euclidean cost is built in
  registers.
- **Analytic backward kernels**, including the adjoint of uDTW's
  soft-selected penalty Ω and dL/dR seeds for JEANIE's accumulator.
- **Exact higher-order derivatives.** The backward is itself an op whose
  derivative uses an autograd-native implementation. Training stays on the
  fast path; any-order and forward-mode derivatives are still exact.

## Development

```bash
uv sync --extra cu126
uv run pytest                                     # tests (CUDA tests auto-skip without a GPU)
bash benchmarks/download_data.sh                  # ECG5000 + NW-UCLA for the real-data benchmarks
uv run python benchmarks/speed_udtw.py            # speed tables
uv run python benchmarks/ecg5000_forecast.py --impl fast --device cuda
```

## Citation

uDTW and JEANIE were introduced in the papers below; please cite them if you use torchwarp.

```bibtex
@inproceedings{wang2022uncertainty,
  title     = {Uncertainty-DTW for Time Series and Sequences},
  author    = {Wang, Lei and Koniusz, Piotr},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {176--195},
  year      = {2022},
  publisher = {Springer}
}

@article{wang2024meet,
  title   = {Meet JEANIE: a Similarity Measure for 3D Skeleton Sequences via Temporal-Viewpoint Alignment},
  author  = {Wang, Lei and Liu, Jun and Zheng, Liang and Gedeon, Tom and Koniusz, Piotr},
  journal = {International Journal of Computer Vision},
  volume  = {132},
  number  = {9},
  pages   = {4091--4122},
  year    = {2024},
  publisher = {Springer}
}
```

## License

[MIT](LICENSE). The uDTW and JEANIE methods are by their authors (see Citation).
