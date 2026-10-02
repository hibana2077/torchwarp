# Real-data check: torchwarp vs the reference implementations

Measured on 2026-10-01 with a TITAN RTX, PyTorch 2.14.1+cu126, and 10 CPU
threads.

In each comparison, only the uDTW or JEANIE module differs between the
reference (github.com/LeiWangR/uDTW, github.com/LeiWangR/JEANIE) and
torchwarp ("fast" in the result files). Everything else is identical: data split, seeds,
initial weights, batch and episode order, hyperparameters and the
evaluation code. Run `python benchmarks/summarize.py` to regenerate every
number below from `benchmarks/results/*.json`. Run
`bash benchmarks/download_data.sh` to fetch the datasets.

## Conclusion

- **The optimised implementations score the same as the reference.**
  - In **float64**, the two implementations give identical results over a
    full training run and its evaluation. The metrics differ by 1e-13 to
    1e-11, and the JEANIE accuracy is exactly equal (40.4133% for all three
    setups).
  - In **float32**, per-seed results differ slightly in both directions,
    with no significant difference (paired t-test p = 0.48–0.86). The same
    optimised code differs just as much between CPU and CUDA, so the spread
    comes from float32 rounding amplified by training, not from the
    implementation.
- **Speed:** a training run is 366× faster for uDTW and 416× faster for
  JEANIE (CUDA vs the reference on CPU). The reference runs fastest on CPU,
  so that is the setup it is compared in.

## 1. uDTW: ECG5000 forecasting (uDTW paper, Sec. 4.3 / Table 2 setup)

The setup:

- **Data:** UCR ECG5000, using the predefined split of 500 train and 4500
  test series. Inputs are the first 84 steps; targets are the last 56.
- **Model:** MLP 84→256→56 plus a SigmaNet (56→64→56,
  σ = 1.5·sigmoid + 0.5).
- **Loss:** d_uDTW + β·Ω with γ=1, β=1, normalize=False.
- **Training:** Adam, lr 1e-3, batch 50, 100 epochs (1000 steps), 5 seeds
  (0–4).
- **Metrics:** a shared float64 evaluator computes MSE, exact DTW, and the
  soft-DTW divergence with γ=1. All three are lower-is-better.

```bash
python benchmarks/ecg5000_forecast.py --impl ref  --device cpu  --seeds 0 1 2 3 4
python benchmarks/ecg5000_forecast.py --impl fast --device cpu  --seeds 0 1 2 3 4
python benchmarks/ecg5000_forecast.py --impl fast --device cuda --seeds 0 1 2 3 4
```

float32 results, mean ± std over 5 seeds:

| metric (lower is better) | reference CPU | fast CPU | fast CUDA |
| --- | --- | --- | --- |
| MSE | 0.4271 ± 0.0179 | 0.4270 ± 0.0177 | 0.4270 ± 0.0177 |
| DTW | 5.1529 ± 0.3632 | 5.1586 ± 0.3650 | 5.1567 ± 0.3790 |
| sDTW div. | 7.9478 ± 0.5453 | 7.9531 ± 0.5429 | 7.9528 ± 0.5532 |
| train time / run | 877.9 s | 81.2 s (10.8×) | 2.40 s (366×) |

- **Paired differences (fast − reference):**
  - MSE: −1.2e-4 (p=0.70)
  - DTW: +5.7e-3 (p=0.49)
  - sDTW div.: +5.2e-3 (p=0.54)
- **Noise floor:** fast CPU vs fast CUDA, which is the same code on two
  devices, differs by up to 0.039 in DTW on a single seed. That is larger
  than the largest reference-vs-fast gap (0.027).

**float64, seeds 0–1** (`--dtype float64`): MSE, DTW, sDTW div. and the
final training loss agree to within 6e-13 for all three setups. For example,
DTW is 4.824513 / 5.471431 in every setup.

## 2. JEANIE: cross-view few-shot skeleton action recognition (NW-UCLA)

The paper's datasets were not available:

- **NTU-60 and NTU-120** require a registration request.
- **The UWA3D Multiview II** site is offline.
- **Kinetics-skeleton** has no 3D or multi-view data.

This benchmark therefore uses the **public NW-UCLA Multiview 3D skeleton
dataset** as a substitute, with a protocol that mirrors the paper's FSAR
pipeline:

- **Temporal blocks:** sequences are resampled to 32 frames and cut into
  blocks of 8 frames with stride 4, giving T=7.
- **Query viewpoint simulation:** rotations about the vertical axis by
  {−30, −15, 0, 15, 30}°, giving K=5.
- **Encoder:** block MLP 480→256→64.
- **Distance:** JEANIE-1D with γ=0.1, ι=1 and Euclidean base distance.
- **Training:** 300 episodes, 5-way 1-shot on classes {1–5} from views 1
  and 2, with cross-entropy over −distance and Adam at lr 1e-3.
- **Testing:** 300 episodes, 5-way 1-shot on the held-out classes
  {6, 8, 9, 11, 12}. Supports come from views 1 and 2 and queries from view
  3 (cross-view), 5 queries per class.
- **Seeds:** 0–4.

```bash
python benchmarks/nwucla_jeanie_fewshot.py --impl ref  --device cpu  --seeds 0 1 2 3 4
python benchmarks/nwucla_jeanie_fewshot.py --impl fast --device cpu  --seeds 0 1 2 3 4
python benchmarks/nwucla_jeanie_fewshot.py --impl fast --device cuda --seeds 0 1 2 3 4
```

The reference ran seeds 0–2 and seeds 3–4 as two invocations because of a
runtime limit. The configuration was identical.

float32 results, mean ± std over 5 seeds:

| metric | reference CPU | fast CPU | fast CUDA |
| --- | --- | --- | --- |
| accuracy % (higher is better) | 35.31 ± 3.41 | 36.45 ± 2.06 | 35.58 ± 2.63 |
| train time / run | 274.6 s | 2.24 s (123×) | 0.66 s (416×) |
| test time / run | 419.3 s | 1.79 s (234×) | 0.34 s (1233×) |

- **Paired differences (fast − reference):** +1.14 points for fast CPU
  (p=0.50) and +0.27 points for fast CUDA (p=0.86).
- **Per-seed spread is large** (up to 5.6 points) and goes in both
  directions. The same fast code differs by 2.6 points between CPU and CUDA
  on seed 2. With 300 low-temperature training steps, float32 rounding is
  enough to send training down a different path.

**float64, seed 0** (`--dtype float64`): accuracy is exactly 40.4133% for
all three setups, and the final training loss matches to within 1e-11.

## Caveats

- **Five seeds per setting.** The uDTW paper averages 100 runs; the
  reference implementation's speed limited this check to 5.
- **JEANIE was tested on a substitute dataset**, and the absolute accuracy
  is not comparable to the paper. The paper's numbers come from a much
  larger encoder trained for far longer on NTU or Kinetics.
- **The benchmark hyperparameters are my own choices.** The paper's
  appendix with its settings was not available. They are the same for both
  implementations.
