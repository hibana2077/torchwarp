# torchwarp

Faster PyTorch/CUDA version of uncertainty-DTW (uDTW) and JEANIE, with soft-DTW and Free Viewpoint Matching (FVM).

## Usage

Using this respontory needs `uv`, you can use `pip install uv` to get uv.

### Setup

```bash
git clone https://github.com/hibana2077/torchwarp && cd torchwarp
uv sync --extra cu126                      # or --extra cu128 / --extra cpu; pip: pip install -r requirements.txt -e .
```

### Code usage

```python
import torchwarp
d, omega = torchwarp.uDTW(gamma=1.0, normalize=True)(X, Y, sigma_x, sigma_y, beta=1.0)  # [B,N,D], [B,M,D], [B,N,1], [B,M,1]
d = torchwarp.JEANIE(gamma=0.1, max_shift=1)(query, support)                          # [B,K,T,D], [B,U,D]
```

## ECG5000

UCR ECG5000 (500 train / 4500 test series, length 140). An MLP (84 → 256 → 56, ReLU) predicts the last 56 steps from the first 84. Adam, batch size 50, 100 epochs (1000 steps). Frame cost: squared Euclidean.

| training loss | hyperparameters |
| --- | --- |
| Euclidean | lr 1e-3 |
| DTW | lr 1e-3 |
| sDTW div. | γ = 0.001, lr 1e-3 |
| uDTW | γ = 1, β = 1, normalize = True, lr 3e-3; SigmaNet 56 → 64 → 56, σ = 2.0·sigmoid(·) + 0.1 |

Test metrics: MSE (per time step), DTW, sDTW div. (γ = 1) and uDTW (γ = 1, β = 1, normalize = True), all in float64. σ for the uDTW metric comes from one fixed SigmaNet (uDTW loss, σ = 1.5·sigmoid(·) + 0.5, lr 1e-3, seed 100), shared by all models.

| training loss | MSE ↓ | DTW ↓ | sDTW div. ↓ | uDTW ↓ | train time (s) |
| --- | --- | --- | --- | --- | --- |
| Euclidean | 0.2161 ± 0.0065 | 5.5127 ± 0.3189 | 7.8376 ± 0.3005 | 2.5342 ± 0.0911 | 0.77 ± 0.06 |
| DTW | 0.7299 ± 0.1767 | 5.3317 ± 0.4190 | 19.0465 ± 3.5771 | 9.3703 ± 2.7353 | 53.59 ± 0.77 |
| sDTW div. | 0.7248 ± 0.1845 | 5.3684 ± 0.5537 | 19.0471 ± 3.7483 | 9.2619 ± 2.7557 | 2.02 ± 0.12 |
| uDTW | 0.2698 ± 0.0138 | 6.8878 ± 0.4812 | 8.4908 ± 0.4687 | 2.7986 ± 0.1459 | 4.36 ± 0.60 |

DTW training uses a plain PyTorch dynamic program; sDTW div. and uDTW use the torchwarp CUDA kernels.

## NW-UCLA

NW-UCLA Multiview 3D skeletons (10 actions, 20 joints, 3 views). Each sequence is resampled to 32 frames and split into 7 temporal blocks (8 frames, stride 4). Block encoder: MLP 480 → 256 → 64. Query viewpoints are simulated by rotations about the vertical axis. Training: 300 5-way 1-shot episodes on classes {1, 2, 3, 4, 5} (views 1 + 2), cross-entropy over −distance / τ, Adam. Testing: 300 5-way 1-shot episodes on classes {6, 8, 9, 11, 12}, supports from views 1 + 2, 5 queries per class from view 3. Base distance: Euclidean.

| method | viewpoints (deg) | hyperparameters |
| --- | --- | --- |
| sDTW | 0 | γ = 0.001, τ = 1, lr 1e-3 |
| sDTW div. | 0 | γ = 0.001, τ = 1, lr 1e-3 |
| uDTW | 0 | γ = 0.1, β = 3, normalize = True, τ = 1, lr 3e-4; σ head 64 → 32 → 1, σ = 1.5·sigmoid(·) + 0.5 |
| FVM | −60, −30, 0, 30, 60 | γ = 0.1, τ = 3, lr 3e-4 |
| JEANIE | −60, −30, 0, 30, 60 | γ = 1, ι = 2, τ = 10, lr 1e-3 |

| method | accuracy (%) ↑ | train time (s) |
| --- | --- | --- |
| sDTW | 24.41 ± 2.47 | 0.89 ± 0.07 |
| sDTW div. | 24.60 ± 3.56 | 1.65 ± 0.13 |
| uDTW | 24.06 ± 8.43 | 2.04 ± 0.04 |
| FVM | 41.15 ± 0.43 | 0.92 ± 0.15 |
| JEANIE | 37.08 ± 2.37 | 0.67 ± 0.11 |

## Citation

```bibtex
@inproceedings{wang2022uncertainty,
  title={Uncertainty-DTW for Time Series and Sequences}, author={Wang, Lei and Koniusz, Piotr},
  booktitle={European Conference on Computer Vision (ECCV)}, pages={176--195}, year={2022}}
@article{wang2024meet,
  title={Meet JEANIE: a Similarity Measure for 3D Skeleton Sequences via Temporal-Viewpoint Alignment},
  author={Wang, Lei and Liu, Jun and Zheng, Liang and Gedeon, Tom and Koniusz, Piotr},
  journal={International Journal of Computer Vision}, volume={132}, number={9}, pages={4091--4122}, year={2024}}
```

MIT License.
