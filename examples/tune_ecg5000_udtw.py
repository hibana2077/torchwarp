"""Validation search for the uDTW training loss of ecg5000_forecast.py (default setup).

Only uDTW is tuned; the baselines keep their published hyperparameters
(soft-DTW and soft-DTW divergence with gamma = 0.001). Runs use a fixed
400/100 split of the training set; the test set is not used.

  gamma        {0.001, 0.01, 0.1, 1}
  beta         {0.01, 0.1, 1}
  kappa, eta   {1.5, 2} x {0.1, 0.5}   (SigmaNet sigma = kappa * sigmoid(.) + eta)
  lr           {1e-3, 3e-3}
  normalize    {True, False}
  seeds        {0, 1}

Selection: lowest validation DTW, averaged over the seeds; a grid point
where a seed diverged (non-finite metric) is not eligible.

Usage:
  python examples/tune_ecg5000_udtw.py run [--shard 0 --shards 2 --device cuda:0]
  python examples/tune_ecg5000_udtw.py select
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import torch

import ecg5000_forecast as E

OUT = E.RESULTS / "tune_udtw"
GRID = dict(gamma=[0.001, 0.01, 0.1, 1.0], beta=[0.01, 0.1, 1.0], sigma_a=[1.5, 2.0],
            sigma_b=[0.1, 0.5], lr=[1e-3, 3e-3], normalize=[True, False])
SEEDS = [0, 1]


def grid():
    keys = sorted(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]


def name(over, seed):
    return "_".join("{}{}".format(k, v) for k, v in sorted(over.items())) + "_s{}.json".format(seed)


def run(shard, shards, device):
    OUT.mkdir(parents=True, exist_ok=True)
    train, val = E.split_validation(E.load_split("TRAIN"))
    tasks = [(o, s) for o in grid() for s in SEEDS]
    for i, (over, seed) in enumerate(tasks):
        out = OUT / name(over, seed)
        if i % shards != shard or out.exists():
            continue
        cfg = dict(E.config("default", "udtw"), **over)
        f, _, t = E.train_one("udtw", cfg, seed, train, device)
        m = E.evaluate(f, None, val, cfg)
        out.write_text(json.dumps(dict(override=over, seed=seed, config=cfg, train_time_s=t, **m)))
        print(i, over, seed, {k: round(v, 4) for k, v in m.items()}, flush=True)


def select():
    rows = []
    for over in grid():
        res = [json.loads((OUT / name(over, s)).read_text())
               for s in SEEDS if (OUT / name(over, s)).exists()]
        if len(res) < len(SEEDS):
            continue
        mean = {k: sum(r[k] for r in res) / len(res) for k in ("mse", "dtw", "sdtw_div")}
        rows.append((mean["dtw"], over, mean))
    diverged = sum(1 for r in rows if not math.isfinite(r[0]))
    rows = sorted((r for r in rows if math.isfinite(r[0])), key=lambda r: r[0])
    print("{} complete grid points, {} diverged; by validation DTW:".format(
        len(rows) + diverged, diverged))
    for v, over, mean in rows[:10]:
        print("  {:.4f}  {}  {}".format(v, over, {k: round(x, 4) for k, x in mean.items()}))
    if rows:
        print("\nLOSSES['udtw'] =", rows[0][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "select"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    run(args.shard, args.shards, args.device) if args.cmd == "run" else select()


if __name__ == "__main__":
    main()
