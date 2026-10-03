# torchwarp

Faster PyTorch version of uncertainty-DTW (uDTW) and JEANIE.

## Installation

### For pip user

```bash
git clone https://github.com/hibana2077/torchwarp && cd torchwarp
pip install -r requirements.txt -e .
```

### For uv user

```bash
git clone https://github.com/hibana2077/torchwarp && cd torchwarp
uv sync --extra cu126
# or --extra cu128 / --extra cpu;
```

## Usage

### Using uDTW

`uDTW` takes two batches of sequences and their per-frame standard deviations σ (e.g. predicted by a small SigmaNet) and returns the uncertainty-weighted distance and the β-weighted penalty Ω, both of shape `[B]`.

```python
import torch, torch.nn as nn
import torchwarp

X = torch.randn(16, 30, 8, device="cuda", requires_grad=True) # [B, N, D]
Y = torch.randn(16, 40, 8, device="cuda") # [B, M, D]
sigma_net = nn.Sequential(nn.Linear(8, 32),
                          nn.ReLU(),
                          nn.Linear(32, 1),
                          nn.Softplus()
                        ).cuda()

udtw = torchwarp.uDTW(gamma=1.0, normalize=True) # normalize: d(x,y) - [d(x,x) + d(y,y)] / 2
d, omega = udtw(X, Y, sigma_net(X) + 1e-3, sigma_net(Y) + 1e-3, beta=1.0) # sigma: [B, N, 1], [B, M, 1]
loss = (d + omega).mean()
loss.backward()
```

`uDTW` builds the per-pair matrices inside the CUDA kernel. For a custom distance or penalty, the two steps can be run separately: `pairwise_matrices` builds the default `[B, N, M]` matrices (cost `‖x_i − y_j‖² / Σ_ij`, penalty `β · log Σ_ij`, with `Σ_ij = ½(σ_i² + σ'_j²)`), and `udtw_from_matrices` runs the dynamic program on any cost and penalty (no `normalize`).

```python
sx, sy = sigma_net(X) + 1e-3, sigma_net(Y) + 1e-3
cost, penalty, variance = torchwarp.pairwise_matrices(X, Y, sx, sy, beta=1.0)  # each [B, N, M]
d, omega = torchwarp.udtw_from_matrices(cost, penalty, gamma=1.0)            # same as uDTW(gamma=1.0)

# e.g. a cosine distance instead of the squared Euclidean one
cos = 1 - torch.nn.functional.cosine_similarity(X.unsqueeze(2), Y.unsqueeze(1), dim=-1)
d, omega = torchwarp.udtw_from_matrices(cos / variance, penalty, gamma=1.0)
```

### Using JEANIE

`JEANIE` aligns a query observed from K simulated viewpoints with a support sequence, jointly over time and viewpoint. `max_shift` (ι) limits the viewpoint change between neighbouring steps.

```python
query = torch.randn(8, 5, 10, 64, device="cuda", requires_grad=True)   # [B, K, T, D]
support = torch.randn(8, 12, 64, device="cuda")                         # [B, U, D]

jeanie = torchwarp.JEANIE(gamma=0.1, max_shift=1, metric="euclidean")   # metric: euclidean | sqeuclidean | rbf
dist = jeanie(query, support)                                           # [B]
dist, R = jeanie(query, support, return_accumulator=True)               # R: [B, K, T, U], differentiable

query_2d = torch.randn(8, 3, 3, 10, 64, device="cuda")                  # [B, K1, K2, T, D] (azimuth x altitude)
dist_2d = torchwarp.JEANIE(gamma=0.1, max_shift=(1, 1))(query_2d, support)
```

## Results

Mean ± std over seeds 42, 43, 44. Hyperparameters are the defaults in [`examples/`](examples/); run `bash examples/download_data.sh` and then `python examples/ecg5000_forecast.py --loss <loss>` or `python examples/nwucla_fewshot.py --method <method>`.

### Speed

Mean time per training step (ms), official implementation vs torchwarp on the same task and settings. CPU: Intel Core i9-10900K (10 threads); GPU: NVIDIA TITAN RTX.

| method (task) | official CPU | torchwarp CPU | speed-up | official GPU | torchwarp GPU | speed-up |
| --- | --- | --- | --- | --- | --- | --- |
| uDTW (ECG5000) | 957.6 | 84.6 | 11× | 1809.7 | 1.35 | 1345× |
| uDTW (NW-UCLA) | 37.5 | 22.0 | 1.7× | 92.2 | 6.18 | 15× |
| FVM (NW-UCLA) | 157.6 | 5.75 | 27× | 363.9 | 2.93 | 124× |
| JEANIE (NW-UCLA) | 961.8 | 10.03 | 96× | 2362.3 | 2.17 | 1091× |

### Time series – ECG5000

UCR ECG5000: forecast the last 40% of each series from the first 60%. Rows: training loss; columns: test metric (lower is better).

| training loss | MSE | DTW | sDTW div. | uDTW |
| --- | --- | --- | --- | --- |
| Euclidean | 0.2161 ± 0.0065 | 5.5127 ± 0.3189 | 7.8376 ± 0.3005 | 2.5342 ± 0.0911 |
| DTW | 0.7299 ± 0.1767 | 5.3317 ± 0.4190 | 19.0465 ± 3.5771 | 9.3703 ± 2.7353 |
| sDTW div. | 0.7248 ± 0.1845 | 5.3684 ± 0.5537 | 19.0471 ± 3.7483 | 9.2619 ± 2.7557 |
| uDTW | 0.2698 ± 0.0138 | 6.8878 ± 0.4812 | 8.4908 ± 0.4687 | 2.7986 ± 0.1459 |

### Few-shot – NW-UCLA

Cross-view 5-way 1-shot action recognition on NW-UCLA (train classes 1–5, test classes 6, 8, 9, 11, 12; queries from the unseen view 3).

| method | accuracy (%) |
| --- | --- |
| sDTW | 24.41 ± 2.47 |
| sDTW div. | 24.60 ± 3.56 |
| uDTW | 24.06 ± 8.43 |
| FVM | 41.15 ± 0.43 |
| JEANIE | 37.08 ± 2.37 |

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