# torchwarp

[![test](https://github.com/hibana2077/torchwarp/actions/workflows/test.yml/badge.svg)](https://github.com/hibana2077/torchwarp/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.5-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6%20%7C%2012.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**torchwarp** is an accelerated PyTorch implementation of uDTW and JEANIE, providing differentiable sequence alignment with optimized CUDA kernels.

This repository provides implementations of uDTW, JEANIE, and FVM; with usage examples and simple script for time series prediction task and few-shot action recognition.

## Documentation

- [API reference](docs/api.md): Module interfaces, functional APIs, tensor shapes, and parameters.
- [Installation](docs/installation.md): CUDA, uv, and optional dependencies.
- [Usage](docs/usage.md): Usage examples for uDTW, JEANIE, FVM, and Soft-DTW, along with scripts to reproduce the experiments.

## Installation

From the repository root:

```bash
pip install -e .            # or -e ".[plot]" to also install matplotlib for torchwarp.plot
```

torchwarp provides two backends: custom CUDA kernels (JIT-compiled on first use) and a vectorized PyTorch implementation. The CUDA backend requires a CUDA-enabled PyTorch build and a compatible CUDA toolkit (`nvcc`). Without these, torchwarp automatically falls back to the PyTorch backend.

For CUDA-specific PyTorch builds, uv installation, and optional dependencies, see the [installation guide](docs/installation.md).

## Example

Like `torch.nn`, every method is a module (`torchwarp.UDTW`,
`torchwarp.SoftDTW`, `torchwarp.JEANIE`, `torchwarp.FVM`) with a matching
function in `torchwarp.functional`. All inputs have a leading batch
dimension `B`.

### uDTW

```python
import torch
import torchwarp

device = "cuda" if torch.cuda.is_available() else "cpu"

# X: [B, N, D],  sigma_x: [B, N, 1]
# Y: [B, M, D],  sigma_y: [B, M, 1]
X = torch.randn(4, 30, 8, device=device, requires_grad=True)
Y = torch.randn(4, 40, 8, device=device)
# sigma must be positive: shift to [0.5, 1.5)
sigma_x = torch.rand(4, 30, 1, device=device) + 0.5
sigma_y = torch.rand(4, 40, 1, device=device) + 0.5

udtw = torchwarp.UDTW(gamma=1.0, beta=1.0)
d, omega = udtw(X, Y, sigma_x, sigma_y)

(d + omega).mean().backward()
```

where:

- `B` = number of sequence pairs
- `N`, `M` = lengths of the two sequences
- `D` = feature dimension

`sigma_x` and `sigma_y` are positive per-element standard deviations
$\sigma > 0$, which can be provided directly or predicted by a model. `d` is the uDTW distance and `omega` the
uncertainty penalty $\beta\,\Omega$, one value per pair (shape `[B]`); the training
loss adds the two. For `normalize=True`, custom base distances and
`torchwarp.functional.udtw_from_cost`, see the [usage guide](docs/usage.md).

### JEANIE

```python
import torch
import torchwarp

device = "cuda" if torch.cuda.is_available() else "cpu"

# query: [B, K, N, D]
# support: [B, M, D]
query = torch.randn(4, 5, 7, 64, device=device, requires_grad=True)
support = torch.randn(4, 7, 64, device=device, requires_grad=True)

jeanie = torchwarp.JEANIE(gamma=0.1, max_shift=1)
d = jeanie(query, support)   # [B]
d.mean().backward()
```

where:

- `K` = number of matching configurations of the query (here, viewpoints)
- `N`, `M` = lengths of the query and support sequences
- `D` = feature dimension

JEANIE aligns the two sequences while limiting how much the query's
configuration may change between consecutive steps (`max_shift`). In
this example the sequence elements are encoded temporal blocks of 3D
skeleton sequences, and the query is rotated to `K` simulated camera
viewpoints before encoding.

The same call in functional form:

```python
import torchwarp.functional as TF

d = TF.jeanie(query, support, gamma=0.1, max_shift=1)
```

### FVM

FVM takes the same `query` and `support` tensors as JEANIE (see above):

```python
fvm = torchwarp.FVM(gamma=0.1)
d = fvm(query, support)   # [B]
d.mean().backward()
```

The functional form also has a `*_from_cost` variant that takes a
precomputed cost tensor:

```python
import torchwarp.functional as TF

cost = TF.euclidean_cost(query, support)   # [B, K, N, M]
d = TF.fvm_from_cost(cost, gamma=0.1)
```

For two viewpoint axes, other base distances and the full list of
functions, see the [usage guide](docs/usage.md) and the
[API reference](docs/api.md).

## Experiments

### Quantitative results

We evaluate torchwarp on time series forecasting and few-shot skeleton action recognition, comparing task performance across alignment methods and computational efficiency improvements.

#### Time series forecasting (ECG5000)

We evaluate multistep forecasting on UCR ECG5000 using the official train/test split (500/4,500 series) without additional normalization. An MLP (84-256-56, ReLU) predicts the last 40% of each series from the first 60%, trained with Adam (lr=0.001, batch size=50) for 100 epochs under identical settings across losses. Results are reported as mean and std over three seeds (42, 43, 44), using MSE and DTW-based evaluation metrics. Soft-DTW and its divergence use γ=0.001.

| loss | MSE | DTW | sDTW div. | uDTW |
| --- | --- | --- | --- | --- |
| Euclidean | 0.2161 ± 0.0065 | 5.5127 ± 0.3189 | 7.8376 ± 0.3005 | 2.5342 ± 0.0911 |
| DTW | 0.7299 ± 0.1767 | 5.3317 ± 0.4190 | 19.0465 ± 3.5771 | 9.3703 ± 2.7353 |
| sDTW | 0.7265 ± 0.1751 | 5.3920 ± 0.4921 | 18.9393 ± 3.6078 | 9.2567 ± 2.6598 |
| sDTW div. | 0.7248 ± 0.1845 | 5.3684 ± 0.5537 | 19.0471 ± 3.7483 | 9.2619 ± 2.7557 |
| uDTW | 0.4003 ± 0.0742 | 5.2258 ± 0.3640 | 8.1221 ± 0.6425 | 3.2134 ± 0.3950 |

#### Few-shot action recognition (NW-UCLA)

We evaluate cross-view 5-way 1-shot skeleton action recognition on NW-UCLA Multiview Action3D, using the `all_sqe` skeleton data provided by CTR-GCN. Sequences are hip-centered, resampled to 32 frames, and divided into seven overlapping 8-frame blocks. An MLP encoder (480-256-64, ReLU) is trained from scratch with Adam for 300 episodes, comparing sDTW, sDTW divergence, uDTW, FVM, and JEANIE. Five disjoint classes are used for training and testing, with views 1–2 used for training and support samples, and unseen view 3 for test queries. Results are reported as mean and std of top-1 accuracy over three seeds (42, 43, 44), each evaluated on 300 episodes (7,500 queries). The experiment follows the established NW-UCLA cross-view split but uses a custom few-shot protocol and MLP encoder.

| method | accuracy (%) |
| --- | --- |
| sDTW | 24.41 ± 2.47 |
| sDTW div. | 24.60 ± 3.56 |
| uDTW | 24.06 ± 8.43 |
| FVM | 41.15 ± 0.43 |
| JEANIE (1-max shift) | 38.75 ± 1.91 |
| JEANIE (2-max shift) | 37.08 ± 2.37 |

#### Speed-up

Mean training-step time in ms (forward and backward; one 5-way 1-shot
episode for NW-UCLA, a batch of 50 series for ECG5000) of the original
implementations and torchwarp under the same settings. Speed-ups compare
the two on the same device.

| method | task | official CPU | torchwarp CPU | speed-up | official GPU | torchwarp GPU | speed-up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| JEANIE (1-max shift) | NW-UCLA | 886.1 | 7.40 | 120× | 2116.3 | 2.74 | 774× |
| JEANIE (2-max shift) | NW-UCLA | 1017.6 | 11.17 | 91× | 2197.0 | 2.28 | 963× |
| FVM | NW-UCLA | 163.6 | 5.53 | 30× | 386.8 | 3.75 | 103× |
| uDTW | NW-UCLA | 40.4 | 21.4 | 1.9× | 86.2 | 7.06 | 12× |
| uDTW | ECG5000 | 1006.9 | 88.6 | 11× | 1856.0 | 2.18 | 852× |

Hardware, software versions, timing method and experimental settings are in the [examples](docs/usage.md#examples).

### Qualitative results

#### uDTW

Alignment paths of sDTW and uDTW for $\gamma = 0.01$ and $0.1$, and the
uncertainty $\sigma$ on the uDTW path with its histogram, for ECG5000 test
series 204: the forecast of the trained uDTW model (seed 42) vs. the ground
truth.

![sDTW and uDTW paths on ECG5000](assets/udtw_paths_ecg5000.png)

#### FVM and JEANIE

Alignment paths of soft-DTW (per viewpoint), FVM and JEANIE for a query
(view 3) and a support (view 1) of the action donning, with the trained
JEANIE encoder (seed 42):

![soft-DTW, FVM and JEANIE paths on NW-UCLA](assets/viewpoint_paths_nwucla.png)

## Acknowledgements

This faster implementation builds upon the original [uDTW](https://github.com/LeiWangR/uDTW) and [JEANIE](https://github.com/LeiWangR/JEANIE) repositories. We thank the original contributors for making their implementations publicly available.

## Citation

If you find this project useful for your research, please cite the following original papers.

### uDTW

```bibtex
@inproceedings{wang2022uncertainty,
  title={Uncertainty-dtw for time series and sequences},
  author={Wang, Lei and Koniusz, Piotr},
  booktitle={European Conference on Computer Vision},
  pages={176--195},
  year={2022},
  organization={Springer}
}
```

### JEANIE

```bibtex
@article{wang2024meet,
  title={Meet jeanie: a similarity measure for 3d skeleton sequences via temporal-viewpoint alignment},
  author={Wang, Lei and Liu, Jun and Zheng, Liang and Gedeon, Tom and Koniusz, Piotr},
  journal={International Journal of Computer Vision},
  volume={132},
  number={9},
  pages={4091--4122},
  year={2024},
  publisher={Springer}
}
```