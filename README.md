# torchwarp

**torchwarp** is an accelerated PyTorch implementation of [uDTW](https://github.com/LeiWangR/uDTW) and [JEANIE](https://github.com/LeiWangR/JEANIE), providing differentiable sequence alignment with optimized CUDA kernels.

This repository includes uDTW, JEANIE and FVM, together with examples and benchmarks.

## Features

- **uDTW, JEANIE and FVM:** differentiable alignment, with forward values and gradients validated against the original implementations (see `tests/`).
- **CUDA acceleration:** up to 963× faster training steps in the NW-UCLA benchmark (see [Results](#results)).
- **Examples:** few-shot action recognition on NW-UCLA, with optional alignment visualizations.

## Installation

From the repository root:

```bash
pip install -e .
```

CUDA acceleration needs a CUDA-enabled PyTorch build and a CUDA toolkit (`nvcc`) of the same major version; without them torchwarp runs on its PyTorch backend. For CUDA-specific PyTorch builds, uv installation, and optional dependencies, see the [installation guide](docs/installation.md).

## Minimal Example

[TODO: Insert a verified minimal uDTW example.]

### JEANIE and FVM

```python
import torch
from torchwarp import euclidean_cost, jeanie_1d_from_cost, fvm_query_only_1d

device = "cuda" if torch.cuda.is_available() else "cpu"

# query_features:   [K, T, D]
# support_features: [U, D]
query_features = torch.randn(5, 7, 64, device=device, requires_grad=True)
support_features = torch.randn(7, 64, device=device)

cost = euclidean_cost(query_features, support_features)

# JEANIE
d_jeanie = jeanie_1d_from_cost(cost, gamma=0.1, max_shift=1)

# FVM
d_fvm = fvm_query_only_1d(cost, gamma=0.1)

d_jeanie.backward()
```

where:

- `K` = number of query viewpoints
- `T` = number of query temporal blocks
- `U` = number of support temporal blocks
- `D` = feature dimension

The features are encoded temporal blocks of 3D skeleton sequences; the
query is rotated to `K` simulated camera viewpoints before encoding.
`cost` has shape `[K, T, U]`.

For batched inputs and more advanced configurations, including two
viewpoint axes, see the [usage guide](docs/usage.md).

## Results

### Few-shot action recognition (NW-UCLA)

5-way 1-shot cross-view classification accuracy of torchwarp (mean ± std
over seeds 42, 43, 44), and mean training-step time in ms (one episode,
forward and backward) of the original implementations and torchwarp
under the same settings. Speed-ups are measured on the same device;
n/a: not benchmarked against an original implementation.

| method | accuracy (%) | official CPU | torchwarp CPU | speed-up | official GPU | torchwarp GPU | speed-up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sDTW | 24.41 ± 2.47 | n/a | n/a | n/a | n/a | n/a | n/a |
| sDTW div. | 24.60 ± 3.56 | n/a | n/a | n/a | n/a | n/a | n/a |
| uDTW | 24.06 ± 8.43 | 40.4 | 21.4 | 1.9× | 86.2 | 7.06 | 12× |
| FVM | 41.15 ± 0.43 | 163.6 | 5.53 | 30× | 386.8 | 3.75 | 103× |
| JEANIE (1-max shift) | 38.75 ± 1.91 | 886.1 | 7.40 | 120× | 2116.3 | 2.74 | 774× |
| JEANIE (2-max shift) | 37.08 ± 2.37 | 1017.6 | 11.17 | 91× | 2197.0 | 2.28 | 963× |

Alignment paths of soft-DTW (per viewpoint), FVM and JEANIE for a query
(view 3) and a support (view 1) of the action donning, with the trained
JEANIE encoder (seed 42):

![soft-DTW, FVM and JEANIE paths on NW-UCLA](assets/viewpoint_paths_nwucla.png)

Hardware, software versions, timing method and experimental settings are in [benchmarks](docs/benchmarks.md).

### Time series (ECG5000)

[TODO: uDTW on ECG5000.]

## Examples

To run the NW-UCLA experiments of the table (seeds 42, 43, 44 by default;
results are written to `examples/results/`):

```bash
bash examples/download_data.sh
for m in sdtw sdtw_div udtw fvm; do python examples/nwucla_fewshot.py --method $m; done
for s in 1 2; do python examples/nwucla_fewshot.py --method jeanie --max-shift $s; done
```

To draw the soft-DTW, FVM and JEANIE alignment paths for the NW-UCLA
test split:

```bash
python examples/nwucla_fewshot.py --method jeanie --seeds 42 --save-ckpt examples/results/ckpt
python examples/plot_jeanie_paths.py
```

[TODO: ECG5000 with uDTW.]

## Documentation

- [Installation](docs/installation.md): CUDA, uv, and optional dependencies.
- [Usage](docs/usage.md): step-by-step usage of uDTW, JEANIE, FVM and soft-DTW.
- [API reference](docs/api.md): arguments, shapes and return values.
- [Benchmarks](docs/benchmarks.md): experimental setups and results.

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