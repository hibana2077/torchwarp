# torchwarp

[![test](https://github.com/hibana2077/torchwarp/actions/workflows/test.yml/badge.svg)](https://github.com/hibana2077/torchwarp/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.5-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6%20%7C%2012.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**torchwarp** is an accelerated PyTorch implementation of [uDTW](https://github.com/LeiWangR/uDTW) and [JEANIE](https://github.com/LeiWangR/JEANIE), providing differentiable sequence alignment with optimized CUDA kernels.

This repository provides implementations of uDTW, JEANIE, and FVM, along with usage examples and experiments demonstrating their performance.

## Documentation

- [Installation](docs/installation.md): CUDA, uv, and optional dependencies.
- [Usage](docs/usage.md): step-by-step usage of uDTW, JEANIE, FVM and soft-DTW.
- [API reference](docs/api.md): arguments, shapes and return values.
- [Benchmarks](docs/benchmarks.md): experimental setups and results.

## Installation

From the repository root:

```bash
pip install -e .            # or -e ".[plot]" to also install matplotlib for torchwarp.plot
```

CUDA acceleration needs a CUDA-enabled PyTorch build and a CUDA toolkit (`nvcc`) of the same major version; without them torchwarp runs on its PyTorch backend. For CUDA-specific PyTorch builds, uv installation, and optional dependencies, see the [installation guide](docs/installation.md).

## Example

### uDTW

```python
import torch
from torchwarp import UDTW

device = "cuda" if torch.cuda.is_available() else "cpu"

# X: [B, N, D],  sigma_x: [B, N, 1]
# Y: [B, M, D],  sigma_y: [B, M, 1]
X = torch.randn(4, 30, 8, device=device, requires_grad=True)
Y = torch.randn(4, 40, 8, device=device)
# sigma must be positive: shift to [0.5, 1.5)
sigma_x = torch.rand(4, 30, 1, device=device) + 0.5
sigma_y = torch.rand(4, 40, 1, device=device) + 0.5

udtw = UDTW(gamma=1.0)
d, omega = udtw(X, Y, sigma_x, sigma_y, beta=1.0)

(d + omega).mean().backward()
```

where:

- `B` = number of sequence pairs
- `N`, `M` = lengths of the two sequences
- `D` = feature dimension

`sigma_x` and `sigma_y` are the uncertainties $\sigma > 0$ of each time step; in
practice they are predicted from the features by a small network
(SigmaNet in the paper). `d` is the uDTW distance and `omega` the
uncertainty penalty $\beta\,\Omega$, one value per pair (shape `[B]`); the training
loss adds the two. For `normalize=True`, custom base distances and
`torchwarp.udtw_from_cost`, see the [usage guide](docs/usage.md).

### JEANIE

```python
import torch
from torchwarp import euclidean_cost, jeanie_1d_from_cost

device = "cuda" if torch.cuda.is_available() else "cpu"

# query_features:   [K, T, D]
# support_features: [U, D]
query_features = torch.randn(5, 7, 64, device=device, requires_grad=True)
support_features = torch.randn(7, 64, device=device)

cost = euclidean_cost(query_features, support_features)   # [K, T, U]
d_jeanie = jeanie_1d_from_cost(cost, gamma=0.1, max_shift=1)
d_jeanie.backward()
```

### FVM

```python
import torch
from torchwarp import euclidean_cost, fvm_query_only_1d

device = "cuda" if torch.cuda.is_available() else "cpu"

# query_features:   [K, T, D]
# support_features: [U, D]
query_features = torch.randn(5, 7, 64, device=device, requires_grad=True)
support_features = torch.randn(7, 64, device=device)

cost = euclidean_cost(query_features, support_features)   # [K, T, U]
d_fvm = fvm_query_only_1d(cost, gamma=0.1)
d_fvm.backward()
```

In both examples:

- `K` = number of query viewpoints
- `T` = number of query temporal blocks
- `U` = number of support temporal blocks
- `D` = feature dimension

The features are encoded temporal blocks of 3D skeleton sequences; the
query is rotated to `K` simulated camera viewpoints before encoding.
`cost` has shape `[K, T, U]`.

For batched inputs and more advanced configurations, including two
viewpoint axes, see the [usage guide](docs/usage.md).

## Example

### Performance

#### Time series forecasting (ECG5000)

Predict the last 40% of each series from the first 60%; test metrics of
the MLP trained with each loss (mean ± std over seeds 42, 43, 44; lower is
better). soft-DTW and soft-DTW divergence use $\gamma = 0.001$; the uDTW
hyperparameters were selected on a validation split of the training set
([`ecg5000_tune.py`](experiments/ecg5000_tune.py)).

| loss | MSE | DTW | sDTW div. | uDTW |
| --- | --- | --- | --- | --- |
| Euclidean | 0.2161 ± 0.0065 | 5.5127 ± 0.3189 | 7.8376 ± 0.3005 | 2.5342 ± 0.0911 |
| DTW | 0.7299 ± 0.1767 | 5.3317 ± 0.4190 | 19.0465 ± 3.5771 | 9.3703 ± 2.7353 |
| sDTW | 0.7265 ± 0.1751 | 5.3920 ± 0.4921 | 18.9393 ± 3.6078 | 9.2567 ± 2.6598 |
| sDTW div. | 0.7248 ± 0.1845 | 5.3684 ± 0.5537 | 19.0471 ± 3.7483 | 9.2619 ± 2.7557 |
| uDTW | 0.4003 ± 0.0742 | 5.2258 ± 0.3640 | 8.1221 ± 0.6425 | 3.2134 ± 0.3950 |

#### Few-shot action recognition (NW-UCLA)

5-way 1-shot cross-view classification accuracy (mean ± std over seeds
42, 43, 44).

| method | accuracy (%) |
| --- | --- |
| sDTW | 24.41 ± 2.47 |
| sDTW div. | 24.60 ± 3.56 |
| uDTW | 24.06 ± 8.43 |
| FVM | 41.15 ± 0.43 |
| JEANIE (1-max shift) | 38.75 ± 1.91 |
| JEANIE (2-max shift) | 37.08 ± 2.37 |

Alignment paths of soft-DTW (per viewpoint), FVM and JEANIE for a query
(view 3) and a support (view 1) of the action donning, with the trained
JEANIE encoder (seed 42):

![soft-DTW, FVM and JEANIE paths on NW-UCLA](assets/viewpoint_paths_nwucla.png)

### Speed-up

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

Hardware, software versions, timing method and experimental settings are in [benchmarks](docs/benchmarks.md).

## Examples

Short scripts to copy from (CPU or GPU):

```bash
python examples/quickstart_udtw.py     # uDTW as a training loss with a learned uncertainty
python examples/quickstart_jeanie.py   # JEANIE and FVM in a 5-way 1-shot episode
python examples/plot_paths.py          # plot the paths of uDTW and JEANIE (needs matplotlib)
```

Every function in `torchwarp.paths` returns an `Alignment` that can draw
itself, e.g. `a.plot("twoway", X=X, Y=Y)`; see
[Visualization](docs/usage.md#visualization).

## Experiments

To run the NW-UCLA experiments of the table (seeds 42, 43, 44 by default;
results are written to `experiments/results/`):

```bash
bash experiments/download_data.sh
for m in sdtw sdtw_div udtw fvm; do python experiments/nwucla_fewshot.py --method $m; done
for s in 1 2; do python experiments/nwucla_fewshot.py --method jeanie --max-shift $s; done
```

To draw the soft-DTW, FVM and JEANIE alignment paths for the NW-UCLA
test split:

```bash
python experiments/nwucla_fewshot.py --method jeanie --seeds 42 --save-ckpt experiments/results/ckpt
python experiments/nwucla_plot_paths.py
```

To run the ECG5000 experiments of the table, the uDTW validation search
and the timing:

```bash
for l in euclidean dtw sdtw sdtw_div udtw; do python experiments/ecg5000_forecast.py --loss $l; done
python experiments/ecg5000_tune.py run && python experiments/ecg5000_tune.py select
python experiments/ecg5000_speed.py
```

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