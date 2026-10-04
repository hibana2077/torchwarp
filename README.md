# torchwarp

## What is torchwarp

torchwarp is an unofficial, faster PyTorch/CUDA implementation of
uncertainty-DTW (uDTW, ECCV 2022) and JEANIE (IJCV 2024), both proposed
by Lei Wang, Piotr Koniusz et al.

Key features:

* uDTW, JEANIE, Free Viewpoint Matching (FVM) and soft-DTW as
  differentiable PyTorch functions and modules.
* CUDA kernels on the GPU and a vectorised PyTorch backend on the CPU,
  with the same values and gradients as the official code.
* Up to 1345× (uDTW) and 1091× (JEANIE) faster training steps than the
  official code on the GPU, see [Speed-up](#speed-up).
* Plots of the alignment paths of uDTW, soft-DTW, FVM and JEANIE.

## Quick start

### Installation

With pip:

```bash
git clone https://github.com/hibana2077/torchwarp && cd torchwarp
pip install -r requirements.txt -e .
```

With uv (pick `cu126`, `cu128` or `cpu`):

```bash
git clone https://github.com/hibana2077/torchwarp && cd torchwarp
uv sync --extra cu126
```

The plots need matplotlib: `pip install -e ".[vis]"` or
`uv sync --extra cu126 --extra vis`.

### uDTW

uDTW compares two batches of sequences (for example time series), given
an uncertainty σ > 0 for every time step, and returns the uDTW distance
and the uncertainty penalty Ω of each pair.

```python
import torch
import torch.nn as nn
import torchwarp

# two batches of sequences: [batch, length, features]
X = torch.randn(16, 30, 8, device="cuda", requires_grad=True)
Y = torch.randn(16, 40, 8, device="cuda")

# uncertainty sigma per time step: [batch, length, 1], positive
sigma_net = nn.Sequential(
    nn.Linear(8, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Softplus(),
).cuda()
sx = sigma_net(X) + 1e-3
sy = sigma_net(Y) + 1e-3

# d and omega: [batch], one distance and one penalty per pair
udtw = torchwarp.uDTW(gamma=1.0, normalize=True)
d, omega = udtw(X, Y, sx, sy, beta=1.0)
loss = (d + omega).mean()
loss.backward()
```

Plot the sDTW and uDTW paths and the uncertainty Σ on the path:

```python
from torchwarp import plot

fig = plot.udtw_figure(X[:1], Y[:1], sx[:1], sy[:1], gammas=(0.01, 0.1))
fig.savefig("udtw_paths.png")
```

ECG5000, forecast of the trained uDTW model vs. ground truth for one
test series:

![uDTW paths on ECG5000](assets/udtw_paths_ecg5000.png)

### JEANIE

JEANIE compares a query and a support skeleton sequence over time and
camera viewpoint. The query is rotated to K simulated viewpoints and
encoded before it is passed in; JEANIE returns one distance per pair.

```python
# query: encoded blocks of the query seen from 5 viewpoints
# [batch, views, blocks, features]
query = torch.randn(8, 5, 7, 64, device="cuda", requires_grad=True)

# support: encoded blocks of the support, [batch, blocks, features]
support = torch.randn(8, 7, 64, device="cuda")

# dist: [batch]
jeanie = torchwarp.JEANIE(gamma=0.1, max_shift=1)
dist = jeanie(query, support)
dist.sum().backward()
```

Plot the paths of soft-DTW (per view), FVM and JEANIE (one panel per
`max_shift`):

```python
angles = [-60, -30, 0, 30, 60]
fig = plot.viewpoint_figure(query[0], support[0], angles, max_shift=(1, 2))
fig.savefig("viewpoint_paths.png")
```

NW-UCLA, a query and a support sequence of the same action (donning):

![soft-DTW, FVM and JEANIE paths on NW-UCLA](assets/viewpoint_paths_nwucla.png)

The two figures use the models trained by the scripts in
[`examples/`](examples/); `python examples/plot_paths.py` draws them
again. See [docs/usage.md](docs/usage.md) for the definitions of the
inputs and outputs, custom distances, how the viewpoints are simulated,
two viewpoint axes, FVM, and the path tensors behind the plots.

## Results

Mean ± std over seeds 42, 43, 44. Hyperparameters are the defaults in [`examples/`](examples/); run `bash examples/download_data.sh` and then `python examples/ecg5000_forecast.py --loss <loss>` or `python examples/nwucla_fewshot.py --method <method>`.

### Speed-up

Mean time per training step (ms), official implementation vs torchwarp on the same task and settings.

| method (task) | official CPU | torchwarp CPU | speed-up | official GPU | torchwarp GPU | speed-up |
| --- | --- | --- | --- | --- | --- | --- |
| uDTW (ECG5000) | 957.6 | 84.6 | 11× | 1809.7 | 1.35 | 1345× |
| uDTW (NW-UCLA) | 37.5 | 22.0 | 1.7× | 92.2 | 6.18 | 15× |
| FVM (NW-UCLA) | 157.6 | 5.75 | 27× | 363.9 | 2.93 | 124× |
| JEANIE (NW-UCLA) | 961.8 | 10.03 | 96× | 2362.3 | 2.17 | 1091× |

### Time series forecasting (ECG5000)

UCR ECG5000: forecast the last 40% of each series from the first 60%. Rows: training loss; columns: test metric (lower is better).

| training loss | MSE | DTW | sDTW div. | uDTW |
| --- | --- | --- | --- | --- |
| Euclidean | 0.2161 ± 0.0065 | 5.5127 ± 0.3189 | 7.8376 ± 0.3005 | 2.5342 ± 0.0911 |
| DTW | 0.7299 ± 0.1767 | 5.3317 ± 0.4190 | 19.0465 ± 3.5771 | 9.3703 ± 2.7353 |
| sDTW | 0.7265 ± 0.1751 | 5.3920 ± 0.4921 | 18.9393 ± 3.6078 | 9.2567 ± 2.6598 |
| sDTW div. | 0.7248 ± 0.1845 | 5.3684 ± 0.5537 | 19.0471 ± 3.7483 | 9.2619 ± 2.7557 |
| uDTW | 0.2698 ± 0.0138 | 6.8878 ± 0.4812 | 8.4908 ± 0.4687 | 2.7986 ± 0.1459 |

### Action recognition (NW-UCLA)

Cross-view 5-way 1-shot action recognition on NW-UCLA (train classes 1 to 5, test classes 6, 8, 9, 11, 12; queries from the unseen view 3).

| method | accuracy (%) |
| --- | --- |
| sDTW | 24.41 ± 2.47 |
| sDTW div. | 24.60 ± 3.56 |
| uDTW | 24.06 ± 8.43 |
| FVM | 41.15 ± 0.43 |
| JEANIE | 37.08 ± 2.37 |

## Acknowledgement

This faster implementation is based on the official uDTW
(https://github.com/LeiWangR/uDTW) and JEANIE
(https://github.com/LeiWangR/JEANIE) code by Lei Wang and Piotr
Koniusz. We thank the authors for releasing their code.

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