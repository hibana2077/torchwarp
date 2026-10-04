# Benchmarks

## Speed-up

Mean time per training step (ms) of the official implementation and of
torchwarp, on the same task, model and settings. Each entry is the mean
over 30 training steps after 3 warm-up steps: one step is a batch of 50
series for ECG5000 and one 5-way 1-shot episode for NW-UCLA, forward and
backward pass included.

| method (task) | official CPU | torchwarp CPU | speed-up | official GPU | torchwarp GPU | speed-up |
| --- | --- | --- | --- | --- | --- | --- |
| uDTW (ECG5000) | 957.6 | 84.6 | 11× | 1809.7 | 1.35 | 1345× |
| uDTW (NW-UCLA) | 37.5 | 22.0 | 1.7× | 92.2 | 6.18 | 15× |
| FVM (NW-UCLA) | 157.6 | 5.75 | 27× | 363.9 | 2.93 | 124× |
| JEANIE (NW-UCLA) | 961.8 | 10.03 | 96× | 2362.3 | 2.17 | 1091× |

Where the speed comes from:

* **uDTW**: the dynamic program runs along anti-diagonals in one fused
  CUDA kernel, which also builds the cost, penalty and variance on the
  fly instead of storing `[B, N, M]` matrices; the backward pass is
  analytic rather than traced through Python loops.
* **JEANIE and FVM**: the same anti-diagonal scheme over the
  (viewpoint, query, support) volume, with the base distances computed
  inside the kernel; on the CPU, the PyTorch backend vectorises each
  anti-diagonal over the batch and the viewpoints.

## Experiments

These experiments check the implementation on the two tasks of the
papers; they are run with the scripts in [`examples/`](../examples/).
Results are mean ± std over seeds 42, 43 and 44. The hyperparameters are
the defaults at the top of each script.

### Data

```bash
bash examples/download_data.sh
```

downloads UCR ECG5000 (about 10 MB) and the NW-UCLA Multiview skeletons
(about 14 MB) into `examples/data/`.

### Time series forecasting (ECG5000)

UCR ECG5000 has 500 training and 4500 test series of length 140. An MLP
(84 → 256 → 56) predicts the last 40% of each series (56 steps) from
the first 60% (84 steps).

* Training losses (squared Euclidean base distance): Euclidean, DTW,
  soft-DTW, soft-DTW divergence and uDTW (distance plus penalty,
  `normalize=True`, σ from a SigmaNet).
* Test metrics (float64, lower is better): MSE, DTW, soft-DTW divergence
  (γ = 1) and uDTW (γ = 1, β = 1, `normalize=True`). The uDTW metric
  takes σ from one fixed SigmaNet trained with uDTW (seed 100), shared
  by all models.

```bash
python examples/ecg5000_forecast.py --loss euclidean
python examples/ecg5000_forecast.py --loss dtw
python examples/ecg5000_forecast.py --loss sdtw
python examples/ecg5000_forecast.py --loss sdtw_div
python examples/ecg5000_forecast.py --loss udtw
```

Rows: training loss; columns: test metric.

| training loss | MSE | DTW | sDTW div. | uDTW |
| --- | --- | --- | --- | --- |
| Euclidean | 0.2161 ± 0.0065 | 5.5127 ± 0.3189 | 7.8376 ± 0.3005 | 2.5342 ± 0.0911 |
| DTW | 0.7299 ± 0.1767 | 5.3317 ± 0.4190 | 19.0465 ± 3.5771 | 9.3703 ± 2.7353 |
| sDTW | 0.7265 ± 0.1751 | 5.3920 ± 0.4921 | 18.9393 ± 3.6078 | 9.2567 ± 2.6598 |
| sDTW div. | 0.7248 ± 0.1845 | 5.3684 ± 0.5537 | 19.0471 ± 3.7483 | 9.2619 ± 2.7557 |
| uDTW | 0.2698 ± 0.0138 | 6.8878 ± 0.4812 | 8.4908 ± 0.4687 | 2.7986 ± 0.1459 |

### Action recognition (NW-UCLA)

Cross-view 5-way 1-shot action recognition on the NW-UCLA Multiview 3D
skeletons (10 actions, 20 joints, 3 Kinect views).

* Each sequence is resampled to 32 frames and split into temporal blocks
  of 8 frames with stride 4 (7 blocks). An MLP (480 → 256 → 64) encodes
  each block.
* Training: 300 episodes on classes 1 to 5, supports and queries from
  views 1 and 2, cross-entropy over the negative distances divided by a
  temperature.
* Testing: 300 episodes on classes 6, 8, 9, 11 and 12, supports from
  views 1 and 2 and five queries per class from the unseen view 3.
* soft-DTW, soft-DTW divergence and uDTW compare the 0° query with the
  support. FVM and JEANIE use the query rotated about the vertical axis
  to -60°, -30°, 0°, 30° and 60°.

```bash
python examples/nwucla_fewshot.py --method sdtw
python examples/nwucla_fewshot.py --method sdtw_div
python examples/nwucla_fewshot.py --method udtw
python examples/nwucla_fewshot.py --method fvm
python examples/nwucla_fewshot.py --method jeanie
```

| method | accuracy (%) |
| --- | --- |
| sDTW | 24.41 ± 2.47 |
| sDTW div. | 24.60 ± 3.56 |
| uDTW | 24.06 ± 8.43 |
| FVM | 41.15 ± 0.43 |
| JEANIE | 37.08 ± 2.37 |

### Script options

| option | script | description |
| --- | --- | --- |
| `--seeds` | both | Seeds to run (default `42 43 44`). |
| `--device` | both | `cuda` (default when available) or `cpu`. |
| `--out` | both | Output JSON file (default `examples/results/<task>_<method>.json`). |
| `--save-ckpt DIR` | both | Save the trained model of each seed to `DIR`. |

Each run writes the configuration, the PyTorch version, and per seed the
test metrics and the training time to the output JSON.

## Figures

The two path figures of the README come from the seed 42 models:

```bash
python examples/ecg5000_forecast.py --loss udtw --seeds 42 --save-ckpt examples/results/ckpt
python examples/nwucla_fewshot.py --method jeanie --seeds 42 --save-ckpt examples/results/ckpt
python examples/plot_paths.py
```

`plot_paths.py` writes `assets/udtw_paths_ecg5000.png` (test series
`--ecg-index`, default 0) and `assets/viewpoint_paths_nwucla.png`
(dataset indices `--query` and `--support`, default 970 and 1005, two
sequences of the action donning).
