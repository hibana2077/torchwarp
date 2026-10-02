"""ECG5000 forecasting benchmark (uDTW paper, Sec. 4.3 / Table 2 setup).

Given the first 60% of each UCR ECG5000 series (84 steps), an MLP predicts
the remaining 40% (56 steps). It is trained with the uDTW loss
d_uDTW + beta * Omega, with Sigma produced by a SigmaNet (Eq. 8). The
reference implementation (github.com/LeiWangR/uDTW) and torchwarp are
trained with the same data, initial weights, batch order and
hyperparameters. Only the uDTW module differs.

Test metrics (lower is better) are computed by one shared float64 evaluator,
independent of the implementation used for training:
  * MSE
  * DTW: exact hard DTW with squared-Euclidean cost
  * sDTW div.: soft-DTW divergence with gamma=1, computed with the
    reference uDTW at sigma=1 (variance 1, zero penalty), which equals soft-DTW.

Usage:
  python benchmarks/ecg5000_forecast.py --impl ref  --device cpu  --seeds 0 1 2 3 4
  python benchmarks/ecg5000_forecast.py --impl fast --device cuda --seeds 0 1 2 3 4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torchwarp.testing import load_reference

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "data"

CONFIG = dict(
    dataset="UCR ECG5000 (500 train / 4500 test, length 140)",
    input_len=84, output_len=56,
    hidden=256, sigma_hidden=64, sigma_a=1.5, sigma_b=0.5,
    gamma=1.0, beta=1.0, normalize=False,
    optimizer="Adam", lr=1e-3, batch_size=50, epochs=100,
    dtype="float32",
)


def load_split(name, dtype=torch.float32):
    arr = np.loadtxt(DATA / "ECG5000_{}.txt".format(name))
    return torch.tensor(arr[:, 1:], dtype=dtype)  # drop the class label


class Forecaster(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg["input_len"], cfg["hidden"]), nn.ReLU(),
                                 nn.Linear(cfg["hidden"], cfg["output_len"]))

    def forward(self, x):
        return self.net(x)


class SigmaNet(nn.Module):
    """Per-timestep sigma in [b, a + b] from a whole output-length sequence."""

    def __init__(self, cfg):
        super().__init__()
        L = cfg["output_len"]
        self.net = nn.Sequential(nn.Linear(L, cfg["sigma_hidden"]), nn.ReLU(),
                                 nn.Linear(cfg["sigma_hidden"], L))
        self.a, self.b = cfg["sigma_a"], cfg["sigma_b"]

    def forward(self, z):
        return self.a * torch.sigmoid(self.net(z)) + self.b


def make_udtw(impl, cfg):
    if impl == "ref":
        uDTW = load_reference("udtw").uDTW
    else:
        from torchwarp import uDTW
    return uDTW(gamma=cfg["gamma"], normalize=cfg["normalize"])


def train_one(impl, device, seed, cfg, train):
    torch.manual_seed(seed)
    f, sigma = Forecaster(cfg), SigmaNet(cfg)            # identical init across runs
    dtype = getattr(torch, cfg["dtype"])
    f, sigma = f.to(device, dtype), sigma.to(device, dtype)
    crit = make_udtw(impl, cfg)
    opt = torch.optim.Adam(list(f.parameters()) + list(sigma.parameters()), lr=cfg["lr"])
    x_all, y_all = train[:, :cfg["input_len"]], train[:, cfg["input_len"]:]
    gen = torch.Generator().manual_seed(seed)

    step_time, losses = 0.0, []
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(len(train), generator=gen)
        for k in range(0, len(train), cfg["batch_size"]):
            idx = perm[k:k + cfg["batch_size"]]
            x, y = x_all[idx].to(device), y_all[idx].to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = f(x)
            d, p = crit(pred.unsqueeze(-1), y.unsqueeze(-1),
                        sigma(pred).unsqueeze(-1), sigma(y).unsqueeze(-1), beta=cfg["beta"])
            loss = (d + p).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            step_time += time.perf_counter() - t0
            losses.append(loss.item())
    return f.cpu(), step_time, losses


# ----------------------------------------------------------- shared evaluator

def hard_dtw(a, b):
    """Exact DTW, squared-Euclidean cost, batched over the first dim (float64)."""
    B, N = a.shape
    M = b.shape[1]
    C = (a.unsqueeze(2) - b.unsqueeze(1)) ** 2
    R = torch.full((B, N + 1, M + 1), float("inf"), dtype=a.dtype)
    R[:, 0, 0] = 0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            R[:, i, j] = C[:, i - 1, j - 1] + torch.minimum(
                torch.minimum(R[:, i - 1, j], R[:, i, j - 1]), R[:, i - 1, j - 1])
    return R[:, N, M]


def evaluate(f, test, cfg):
    RefUDTW = load_reference("udtw").uDTW

    x = test[:, :cfg["input_len"]].double()
    y = test[:, cfg["input_len"]:].double()
    with torch.no_grad():
        pred = f.double()(x)
        one = torch.ones(len(y), y.shape[1], 1, dtype=torch.float64)
        sdtw_div, _ = RefUDTW(gamma=1.0, normalize=True)(
            pred.unsqueeze(-1), y.unsqueeze(-1), one, one, beta=0.0)
        return dict(
            mse=float(((pred - y) ** 2).mean()),
            dtw=float(hard_dtw(pred, y).mean()),
            sdtw_div=float(sdtw_div.mean()),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["ref", "fast"], required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="smoke tests only")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    if args.epochs is not None:
        CONFIG["epochs"] = args.epochs
    CONFIG["dtype"] = args.dtype
    train = load_split("TRAIN", getattr(torch, args.dtype))
    test = load_split("TEST", getattr(torch, args.dtype))
    out = Path(args.out or ROOT / "benchmarks" / "results" /
               "ecg5000_{}_{}.json".format(args.impl, args.device))
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        f, t, losses = train_one(args.impl, args.device, seed, CONFIG, train)
        metrics = evaluate(f, test, CONFIG)
        runs.append(dict(seed=seed, train_time_s=t, final_train_loss=float(np.mean(losses[-10:])),
                         **metrics))
        print(json.dumps(runs[-1]), flush=True)
        out.write_text(json.dumps(dict(impl=args.impl, device=args.device, config=CONFIG,
                                       torch=torch.__version__, runs=runs), indent=2))


if __name__ == "__main__":
    main()
