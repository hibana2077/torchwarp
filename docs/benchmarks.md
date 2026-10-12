# Benchmarks

## Speed-up

Mean time per training step (ms) of the original implementations and of
torchwarp, on the same task, model and settings.

* **Hardware:** NVIDIA TITAN RTX; Intel Core i9-10900K, 10 CPU threads.
* **Software:** PyTorch 2.14.1 (CUDA 12.6 build), nvcc 12.4.
* **Timing:** mean over 30 training steps after 3 warm-up steps; one step
  is one 5-way 1-shot episode for NW-UCLA (a batch of 50 series for
  ECG5000), forward pass, loss, backward pass and optimizer step
  included. The GPU is synchronized before and after each step. The
  warm-up steps also trigger the one-time JIT compilation of the CUDA
  extensions, so compilation is not included.
* **Same device:** each speed-up compares the original implementation
  and torchwarp on the same device (CPU with CPU, GPU with GPU).

The original implementations run the dynamic program as Python loops
over the cells, with one small tensor operation per cell. On the GPU each
of these operations is a separate kernel launch, so launch overhead
dominates and the original code is slower on the GPU than on the CPU.
torchwarp runs the whole dynamic program in one kernel.

| method (task) | official CPU | torchwarp CPU | speed-up | official GPU | torchwarp GPU | speed-up |
| --- | --- | --- | --- | --- | --- | --- |
| uDTW (ECG5000) | 1006.9 | 88.6 | 11× | 1856.0 | 2.18 | 852× |
| uDTW (NW-UCLA) | 40.4 | 21.4 | 1.9× | 86.2 | 7.06 | 12× |
| FVM (NW-UCLA) | 163.6 | 5.53 | 30× | 386.8 | 3.75 | 103× |
| JEANIE, 1-max shift (NW-UCLA) | 886.1 | 7.40 | 120× | 2116.3 | 2.74 | 774× |
| JEANIE, 2-max shift (NW-UCLA) | 1017.6 | 11.17 | 91× | 2197.0 | 2.28 | 963× |

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
The hyperparameters are the defaults at the top of each script.

### Data

```bash
bash examples/download_data.sh
```

downloads UCR ECG5000 (about 10 MB) and the NW-UCLA Multiview skeletons
(about 14 MB) into `examples/data/`.

### Time series forecasting (ECG5000)

UCR ECG5000 has 500 training and 4500 test series of length 140. An MLP
predicts the last 40% of each series (56 steps) from the first 60% (84
steps). Models are trained on the 500 training series and evaluated on
the 4500 test series.

The training settings stated in the uDTW paper
([arXiv 2211.00005](https://arxiv.org/abs/2211.00005), App. H.2, H.5
and I.2) are available as `--setup paper`:

* MLP 84 → t′ → 56 with a tanh in between; every weight, bias included,
  drawn from N(0, 1).
* SGD, learning rate 1e-3, weight decay 1e-6.
* Per-frame SigmaNet σ_t = κ · sigmoid(FC(ψ_t)) + η with κ = 1.5 and
  η = 0.01 (one value per frame, as in App. H.2 and the official
  `SimpleSigmaNet`), and Σ = 0.5 (σ² + σ′²) (Eq. 8).
* Training losses (squared Euclidean base distance): Euclidean, DTW,
  soft-DTW, soft-DTW divergence and uDTW (d_uDTW + β Ω, without
  normalisation).

The paper does not give t′, the number of epochs, the batch size, γ or
β for this task, and tunes the remaining hyperparameters on a validation
set. [`examples/tune_ecg5000_paper.py`](../examples/tune_ecg5000_paper.py)
does the same on a fixed 400/100 split of the training set (the test set
is not used): t′ ∈ {30, 50}, batch size ∈ {1, 10, 50}, epochs ∈ {30, 50,
100}, γ ∈ {0.01, 0.1, 1}, β ∈ {0.01, 0.1, 1}, two seeds per point.
Euclidean is selected by validation MSE and the alignment losses by
validation DTW; grid points where a seed diverged are skipped. The
selected values go into `PAPER_LOSSES` in
[`examples/ecg5000_forecast.py`](../examples/ecg5000_forecast.py).

Test metrics (float64, lower is better): MSE (per time step), SSE
(summed over the 56 predicted steps), DTW, soft-DTW divergence (γ = 1)
and uDTW (γ = 1, β = 1, `normalize=True`). The uDTW metric takes σ from
one fixed SigmaNet trained with the uDTW setting (seed 100), shared by
all models.

```bash
python examples/tune_ecg5000_paper.py count            # 204 grid runs
python examples/tune_ecg5000_paper.py run --task 0     # one grid run
python examples/tune_ecg5000_paper.py select           # pick PAPER_LOSSES

python examples/ecg5000_forecast.py --setup paper --loss euclidean --seeds $(seq 0 99)
python examples/ecg5000_forecast.py --setup paper --loss dtw --seeds $(seq 0 99)
python examples/ecg5000_forecast.py --setup paper --loss sdtw --seeds $(seq 0 99)
python examples/ecg5000_forecast.py --setup paper --loss sdtw_div --seeds $(seq 0 99)
python examples/ecg5000_forecast.py --setup paper --loss udtw --seeds $(seq 0 99)
```

The previous default setting (ReLU MLP 84 → 256 → 56, PyTorch's default
init, Adam) is still available as `--setup default`.

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
python examples/nwucla_fewshot.py --method jeanie --max-shift 1
python examples/nwucla_fewshot.py --method jeanie --max-shift 2
```

| method | accuracy (%) |
| --- | --- |
| sDTW | 24.41 ± 2.47 |
| sDTW div. | 24.60 ± 3.56 |
| uDTW | 24.06 ± 8.43 |
| FVM | 41.15 ± 0.43 |
| JEANIE (1-max shift) | 38.75 ± 1.91 |
| JEANIE (2-max shift) | 37.08 ± 2.37 |

### Script options

| option | script | description |
| --- | --- | --- |
| `--seeds` | both | Seeds to run (default `42 43 44`). |
| `--setup` | ECG5000 | `default` or `paper` (the uDTW paper's pipeline). |
| `--val` | ECG5000 | Train on 400 training series and evaluate on the other 100. |
| `--set KEY=VALUE` | ECG5000 | Override configuration entries, e.g. `--set epochs=30 gamma=0.1`. |
| `--eval-epochs` | ECG5000 | Also evaluate after these epochs. |
| `--threads` | ECG5000 | Number of CPU threads. |
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

To draw the soft-DTW, FVM and JEANIE paths for the whole NW-UCLA test
split (225 queries of view 3, each with a random support of the same
action from views 1 and 2):

```bash
python examples/plot_jeanie_paths.py
```

The figures and `index.csv` (the distances in each figure) are written to
`examples/results/figures/nwucla/`. Options: `--support nearest` uses the
same-action support with the smallest JEANIE distance, `--max-shift`
sets the JEANIE panels (default `1 2`), `--limit N` draws only the first
N queries, and `--query I --support J` draws one pair.

