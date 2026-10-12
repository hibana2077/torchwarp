"""Validation searches for ecg5000_forecast.py, one per setup (--setup).

Both use a fixed 400/100 split of the training set; the test set is not used.
A grid point where a seed diverged (non-finite metric) is not eligible.

default: only the uDTW loss is tuned; the baselines keep their published
hyperparameters (soft-DTW and soft-DTW divergence with gamma = 0.001).

  gamma        {0.001, 0.01, 0.1, 1}
  beta         {0.01, 0.1, 1}
  kappa, eta   {1.5, 2} x {0.1, 0.5}   (SigmaNet sigma = kappa * sigmoid(.) + eta)
  lr           {1e-3, 3e-3}
  normalize    {True, False}
  seeds        {0, 1}

  Selection: lowest validation DTW, averaged over the seeds.

paper: the uDTW paper fixes the architecture, init, optimizer and SigmaNet
(App. H) but not t', the number of epochs, the batch size, gamma or beta; it
tunes the remaining hyperparameters on a validation set (App. I). Every loss
is tuned:

  hidden (t')  {30, 50}
  batch size   {1, 10, 50}
  epochs       {30, 50, 100}        (one run, evaluated after each)
  gamma        {0.01, 0.1, 1}       (sdtw, sdtw_div, udtw)
  beta         {0.01, 0.1, 1}       (udtw)
  seeds        {0, 1}

  Selection: Euclidean by validation MSE, every alignment loss by validation
  DTW (exact, parameter-free shape metric), averaged over the seeds.

Usage:
  python experiments/ecg5000_tune.py run [--shard 0 --shards 2 --device cuda:0]
  python experiments/ecg5000_tune.py select                          # LOSSES['udtw']
  python experiments/ecg5000_tune.py --setup paper count             # 204 grid runs
  python experiments/ecg5000_tune.py --setup paper run --task 17     # one grid run
  python experiments/ecg5000_tune.py --setup paper select            # PAPER_LOSSES
"""

import argparse
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

import ecg5000_forecast as E

# ---------------------------------------------------------------- default setup

DEFAULT_OUT = E.RESULTS / "tune_udtw"
DEFAULT_GRID = dict(gamma=[0.001, 0.01, 0.1, 1.0], beta=[0.01, 0.1, 1.0], sigma_a=[1.5, 2.0],
                    sigma_b=[0.1, 0.5], lr=[1e-3, 3e-3], normalize=[True, False])
DEFAULT_SEEDS = [0, 1]


def default_grid():
    keys = sorted(DEFAULT_GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(DEFAULT_GRID[k] for k in keys))]


def default_name(over, seed):
    return "_".join("{}{}".format(k, v) for k, v in sorted(over.items())) + "_s{}.json".format(seed)


def default_run(shard, shards, device):
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    train, val = E.split_validation(E.load_split("TRAIN"))
    tasks = [(o, s) for o in default_grid() for s in DEFAULT_SEEDS]
    for i, (over, seed) in enumerate(tasks):
        out = DEFAULT_OUT / default_name(over, seed)
        if i % shards != shard or out.exists():
            continue
        cfg = dict(E.config("default", "udtw"), **over)
        f, _, t = E.train_one("udtw", cfg, seed, train, device)
        m = E.evaluate(f, None, val, cfg)
        out.write_text(json.dumps(dict(override=over, seed=seed, config=cfg, train_time_s=t, **m)))
        print(i, over, seed, {k: round(v, 4) for k, v in m.items()}, flush=True)


def default_select():
    rows = []
    for over in default_grid():
        res = [json.loads((DEFAULT_OUT / default_name(over, s)).read_text())
               for s in DEFAULT_SEEDS if (DEFAULT_OUT / default_name(over, s)).exists()]
        if len(res) < len(DEFAULT_SEEDS):
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


# ------------------------------------------------------------------ paper setup

PAPER_OUT = E.RESULTS / "tune_paper"
HIDDEN, BATCH, EPOCHS = [30, 50], [1, 10, 50], [30, 50, 100]
GAMMA, BETA, PAPER_SEEDS = [0.01, 0.1, 1.0], [0.01, 0.1, 1.0], [0, 1]
SELECT_BY = dict(euclidean="mse", dtw="dtw", sdtw="dtw", sdtw_div="dtw", udtw="dtw")


def paper_grid():
    tasks = []
    for loss in ["euclidean", "dtw", "sdtw", "sdtw_div", "udtw"]:
        gammas = GAMMA if loss in ("sdtw", "sdtw_div", "udtw") else [None]
        betas = BETA if loss == "udtw" else [None]
        for h, bs, g, b, seed in itertools.product(HIDDEN, BATCH, gammas, betas, PAPER_SEEDS):
            cfg = dict(hidden=h, batch_size=bs, epochs=max(EPOCHS))
            if g is not None:
                cfg["gamma"] = g
            if b is not None:
                cfg["beta"] = b
            tasks.append((loss, cfg, seed))
    # most expensive first, so the long jobs start early
    cost = dict(euclidean=1, dtw=2, sdtw=3, sdtw_div=7, udtw=5)
    tasks.sort(key=lambda t: -cost[t[0]] / t[1]["batch_size"])
    return tasks


def paper_name(loss, cfg, seed):
    return "{}_{}_s{}".format(loss, "_".join("{}{}".format(k, v) for k, v in sorted(cfg.items())), seed)


def paper_run(task, threads):
    loss, cfg, seed = paper_grid()[task]
    out = PAPER_OUT / (paper_name(loss, cfg, seed) + ".json")
    if out.exists():
        return
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(Path(E.__file__).resolve()), "--setup", "paper",
           "--loss", loss, "--val", "--device", "cpu", "--threads", str(threads),
           "--seeds", str(seed), "--eval-epochs", *map(str, EPOCHS), "--out", str(out) + ".tmp",
           "--set", *["{}={}".format(k, v) for k, v in cfg.items()]]
    subprocess.run(cmd, check=True)
    Path(str(out) + ".tmp").rename(out)


def paper_select():
    scores = {}                                   # (loss, cfg-with-epochs) -> {seed: metrics}
    for path in PAPER_OUT.glob("*.json"):
        res = json.loads(path.read_text())
        r = res["runs"][0]
        keys = ["hidden", "batch_size"]
        keys += ["gamma"] if res["loss"] in ("sdtw", "sdtw_div", "udtw") else []
        keys += ["beta"] if res["loss"] == "udtw" else []
        base = {k: res["config"][k] for k in keys}
        for ep in EPOCHS:
            m = r if ep == res["config"]["epochs"] else r["at_epoch"][str(ep)]
            key = (res["loss"], json.dumps(dict(base, epochs=ep), sort_keys=True))
            scores.setdefault(key, {})[r["seed"]] = m
    chosen = {}
    for loss, metric in SELECT_BY.items():
        rows = [(sum(m[metric] for m in s.values()) / len(s), cfg, s)
                for (l, cfg), s in scores.items() if l == loss and len(s) == len(PAPER_SEEDS)]
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
    ap.add_argument("--setup", choices=["default", "paper"], default="default")
    ap.add_argument("cmd", choices=["count", "run", "select"])
    ap.add_argument("--shard", type=int, default=0, help="default setup")
    ap.add_argument("--shards", type=int, default=1, help="default setup")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="default setup")
    ap.add_argument("--task", type=int, nargs="*", default=[], help="paper setup")
    ap.add_argument("--threads", type=int, default=1, help="paper setup")
    args = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    if args.setup == "default":
        if args.cmd == "count":
            print(len(default_grid()) * len(DEFAULT_SEEDS))
        elif args.cmd == "run":
            default_run(args.shard, args.shards, args.device)
        else:
            default_select()
    else:
        if args.cmd == "count":
            print(len(paper_grid()))
        elif args.cmd == "run":
            for t in args.task:
                paper_run(t, args.threads)
        else:
            paper_select()


if __name__ == "__main__":
    main()
