"""Validation search for the ECG5000 paper setup (ecg5000_forecast.py --setup paper).

The uDTW paper fixes the architecture, init, optimizer and SigmaNet (App. H)
but not t', the number of epochs, the batch size, gamma or beta; it tunes the
remaining hyperparameters on a validation set (App. I). This script runs that
search on a fixed 400/100 split of the training set:

  hidden (t')  {30, 50}
  batch size   {1, 10, 50}
  epochs       {30, 50, 100}        (one run, evaluated after each)
  gamma        {0.01, 0.1, 1}       (sdtw, sdtw_div, udtw)
  beta         {0.01, 0.1, 1}       (udtw)
  grad clip    {off, max norm 10}   (all losses)
  kappa, eta   {1.5, 3} x {0.01, 0.1, 0.5}   (udtw SigmaNet; the paper's 1.5 and
                                    0.01 were chosen for action recognition)
  seeds        {0, 1}

Selection: Euclidean by validation MSE, every alignment loss by validation
DTW (exact, parameter-free shape metric), averaged over the seeds. A grid
point where any seed diverged (non-finite metric) is not eligible.

Usage:
  python examples/tune_ecg5000_paper.py count
  python examples/tune_ecg5000_paper.py run --task 17          # one grid point
  python examples/tune_ecg5000_paper.py select                 # pick configs
"""

import argparse
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "results" / "tune_paper"

HIDDEN, BATCH, EPOCHS = [30, 50], [1, 10, 50], [30, 50, 100]
GAMMA, BETA, SEEDS = [0.01, 0.1, 1.0], [0.01, 0.1, 1.0], [0, 1]
CLIP, KAPPA, ETA = [0.0, 10.0], [1.5, 3.0], [0.01, 0.1, 0.5]
DEFAULT = dict(clip=0.0, sigma_a=1.5, sigma_b=0.01)       # PAPER_BASE values
SELECT_BY = dict(euclidean="mse", dtw="dtw", sdtw="dtw", sdtw_div="dtw", udtw="dtw")


def grid():
    tasks = []
    for loss in ["euclidean", "dtw", "sdtw", "sdtw_div", "udtw"]:
        gammas = GAMMA if loss in ("sdtw", "sdtw_div", "udtw") else [None]
        betas = BETA if loss == "udtw" else [None]
        sigmas = itertools.product(KAPPA, ETA) if loss == "udtw" else [(None, None)]
        for (k, e), clip, h, bs, g, b, seed in itertools.product(
                list(sigmas), CLIP, HIDDEN, BATCH, gammas, betas, SEEDS):
            cfg = dict(hidden=h, batch_size=bs, epochs=max(EPOCHS))
            extra = dict(gamma=g, beta=b, clip=clip, sigma_a=k, sigma_b=e)
            # values equal to PAPER_BASE are left out, so file names stay stable
            cfg.update({key: v for key, v in extra.items()
                        if v is not None and DEFAULT.get(key) != v})
            tasks.append((loss, cfg, seed))
    # most expensive first, so the long jobs start early
    cost = dict(euclidean=1, dtw=2, sdtw=3, sdtw_div=7, udtw=5)
    tasks.sort(key=lambda t: -cost[t[0]] / t[1]["batch_size"])
    return tasks


def name(loss, cfg, seed):
    return "{}_{}_s{}".format(loss, "_".join("{}{}".format(k, v) for k, v in sorted(cfg.items())), seed)


def run(task, threads):
    loss, cfg, seed = grid()[task]
    out = OUT / (name(loss, cfg, seed) + ".json")
    if out.exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "examples" / "ecg5000_forecast.py"), "--setup", "paper",
           "--loss", loss, "--val", "--device", "cpu", "--threads", str(threads),
           "--seeds", str(seed), "--eval-epochs", *map(str, EPOCHS), "--out", str(out) + ".tmp",
           "--set", *["{}={}".format(k, v) for k, v in cfg.items()]]
    subprocess.run(cmd, check=True)
    Path(str(out) + ".tmp").rename(out)


def select():
    scores = {}                                   # (loss, cfg-with-epochs) -> {seed: metrics}
    for path in OUT.glob("*.json"):
        res = json.loads(path.read_text())
        r = res["runs"][0]
        keys = ["hidden", "batch_size"]
        keys += ["gamma"] if res["loss"] in ("sdtw", "sdtw_div", "udtw") else []
        keys += ["beta", "sigma_a", "sigma_b"] if res["loss"] == "udtw" else []
        base = {k: res["config"][k] for k in keys}
        base["clip"] = res["config"].get("clip", DEFAULT["clip"])
        for ep in EPOCHS:
            m = r if ep == res["config"]["epochs"] else r["at_epoch"][str(ep)]
            key = (res["loss"], json.dumps(dict(base, epochs=ep), sort_keys=True))
            scores.setdefault(key, {})[r["seed"]] = m
    chosen = {}
    for loss, metric in SELECT_BY.items():
        rows = [(sum(m[metric] for m in s.values()) / len(s), cfg, s)
                for (l, cfg), s in scores.items() if l == loss and len(s) == len(SEEDS)]
        total = sum(1 for (l, _) in scores if l == loss)
        diverged = sum(1 for r in rows if not math.isfinite(r[0]))
        rows = [r for r in rows if math.isfinite(r[0])]
        if not rows:
            print("{}: no complete, finite grid points".format(loss))
            continue
        rows.sort(key=lambda r: r[0])
        print("{} ({} finite, {} diverged, of {} grid points; by val {}):".format(
            loss, len(rows), diverged, total, metric))
        for v, cfg, s in rows[:5]:
            mean = {k: sum(m[k] for m in s.values()) / len(s) for k in ("mse", "sse", "dtw", "sdtw_div")}
            print("  {:.4f}  {}  {}".format(v, cfg, {k: round(x, 3) for k, x in mean.items()}))
        chosen[loss] = json.loads(rows[0][1])
    print("\nPAPER_LOSSES =", json.dumps(chosen, indent=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["count", "run", "select"])
    ap.add_argument("--task", type=int, nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "count":
        print(len(grid()))
    elif args.cmd == "run":
        for t in args.task:
            run(t, args.threads)
    else:
        select()


if __name__ == "__main__":
    main()
