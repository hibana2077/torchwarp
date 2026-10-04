"""ECG5000 forecasting (UCR ECG5000, 500 train / 4500 test series of length 140).

An MLP (84 -> 256 -> 56) predicts the last 40% of each series from the first
60%. Training losses (squared-Euclidean frame cost):
  * euclidean: sum_t (pred_t - y_t)^2
  * dtw:       hard DTW (plain PyTorch dynamic program)
  * sdtw:      soft-DTW sdtw(x,y)
  * sdtw_div:  soft-DTW divergence sdtw(x,y) - [sdtw(x,x) + sdtw(y,y)] / 2
  * udtw:      d_uDTW + beta * Omega (normalize=True), sigma from a SigmaNet

Test metrics (float64, lower is better): MSE, DTW, sDTW div. (gamma=1) and
uDTW (gamma=1, beta=1, normalize=True). The uDTW metric takes sigma from one
fixed evaluation SigmaNet (a uDTW run with the base configuration and seed
EVAL_SEED), shared by every model.

Usage:
  python examples/ecg5000_forecast.py --loss udtw
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchwarp import soft_dtw, uDTW

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
RESULTS = ROOT / "examples" / "results"

BASE = dict(
    input_len=84, output_len=56, hidden=256,
    sigma_hidden=64, sigma_a=1.5, sigma_b=0.5,   # sigma = a * sigmoid(.) + b
    gamma=1.0, beta=1.0, normalize=True,
    lr=1e-3, batch_size=50, epochs=100,
)
LOSSES = {
    "euclidean": {},
    "dtw": {},
    "sdtw": dict(gamma=0.001),
    "sdtw_div": dict(gamma=0.001),
    "udtw": dict(gamma=1.0, beta=1.0, sigma_a=2.0, sigma_b=0.1, lr=3e-3),
}
EVAL_SEED = 100


def load_split(name):
    arr = np.loadtxt(DATA / "ECG5000_{}.txt".format(name))
    return torch.tensor(arr[:, 1:], dtype=torch.float32)  # drop the class label


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


def hard_dtw_loss(a, b):
    """Hard DTW, squared-Euclidean cost, differentiable (sub-gradient of min).

    a [B,N], b [B,M] -> [B]. Anti-diagonal sweep: diagonal d holds R[i, d-i]
    for i = 0..N of the (N+1)x(M+1) accumulated-cost table.
    """
    B, N = a.shape
    M = b.shape[1]
    inf = torch.tensor(float("inf"), dtype=a.dtype, device=a.device)
    C = F.pad((a.unsqueeze(2) - b.unsqueeze(1)) ** 2, (1, 1, 1, 0), value=float("inf"))
    i = torch.arange(N + 1, device=a.device)                 # C row i-1 <-> padded row i

    def shift(v):                                            # v[i] -> v[i-1], inf at i=0
        return torch.cat([inf.expand(B, 1), v[:, :-1]], 1)

    prev2 = torch.full((B, N + 1), float("inf"), dtype=a.dtype, device=a.device)
    prev1 = prev2.clone()
    prev1[:, 0] = 0                                          # d = 0: R[0,0] = 0
    for d in range(1, N + M + 1):
        j = (d - i).clamp(0, M + 1)                          # padded column index
        cd = C[:, i, j]                                      # C[i-1, j-1], inf off-grid
        cur = cd + torch.minimum(torch.minimum(shift(prev1), prev1), shift(prev2))
        prev2, prev1 = prev1, cur
    return prev1[:, N]


def make_loss(name, cfg):
    """loss(pred, y, sigma) -> scalar, averaged over the batch."""
    if name == "euclidean":
        return lambda pred, y, sigma: ((pred - y) ** 2).sum(1).mean()
    if name == "dtw":
        return lambda pred, y, sigma: hard_dtw_loss(pred, y).mean()
    if name in ("sdtw", "sdtw_div"):
        def sdtw(a, b):
            return soft_dtw((a.unsqueeze(2) - b.unsqueeze(1)) ** 2, cfg["gamma"])
        if name == "sdtw":
            return lambda pred, y, sigma: sdtw(pred, y).mean()
        return lambda pred, y, sigma: (sdtw(pred, y) - 0.5 * (sdtw(pred, pred) + sdtw(y, y))).mean()
    crit = uDTW(gamma=cfg["gamma"], normalize=cfg["normalize"])

    def loss(pred, y, sigma):
        d, p = crit(pred.unsqueeze(-1), y.unsqueeze(-1),
                    sigma(pred).unsqueeze(-1), sigma(y).unsqueeze(-1), beta=cfg["beta"])
        return (d + p).mean()
    return loss


def train_one(name, cfg, seed, train, device):
    torch.manual_seed(seed)
    f, sigma = Forecaster(cfg).to(device), SigmaNet(cfg).to(device)
    crit = make_loss(name, cfg)
    params = list(f.parameters()) + (list(sigma.parameters()) if name == "udtw" else [])
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    x_all, y_all = train[:, :cfg["input_len"]], train[:, cfg["input_len"]:]
    gen = torch.Generator().manual_seed(seed)

    step_time = 0.0
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(len(train), generator=gen)
        for k in range(0, len(train), cfg["batch_size"]):
            idx = perm[k:k + cfg["batch_size"]]
            x, y = x_all[idx].to(device), y_all[idx].to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = crit(f(x), y, sigma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            step_time += time.perf_counter() - t0
    return f.cpu(), sigma.cpu(), step_time


# ------------------------------------------------------------------ evaluation

def hard_dtw(a, b):
    """Hard DTW, squared-Euclidean cost, batched over the first dim."""
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


def eval_sigmanet(train, device):
    """The fixed SigmaNet of the uDTW test metric (trained once, then cached)."""
    path = RESULTS / "ecg5000_eval_sigmanet_seed{}.pt".format(EVAL_SEED)
    sigma = SigmaNet(BASE)
    if path.exists():
        sigma.load_state_dict(torch.load(path))
    else:
        _, sigma, _ = train_one("udtw", BASE, EVAL_SEED, train, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sigma.state_dict(), path)
    return sigma.double().eval()


def evaluate(f, sigma_eval, test, cfg):
    metric = uDTW(gamma=1.0, normalize=True, backend="torch")
    x = test[:, :cfg["input_len"]].double()
    y = test[:, cfg["input_len"]:].double()
    with torch.no_grad():
        pred = f.double()(x)
        p3, y3 = pred.unsqueeze(-1), y.unsqueeze(-1)
        one = torch.ones_like(y3)
        sdtw_div, _ = metric(p3, y3, one, one, beta=0.0)      # sigma = 1: soft-DTW
        d, pen = metric(p3, y3, sigma_eval(pred).unsqueeze(-1), sigma_eval(y).unsqueeze(-1),
                        beta=1.0)
        return dict(
            mse=float(((pred - y) ** 2).mean()),
            dtw=float(hard_dtw(pred, y).mean()),
            sdtw_div=float(sdtw_div.mean()),
            udtw=float((d + pen).mean()),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=sorted(LOSSES), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-ckpt", default=None, metavar="DIR")
    args = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    cfg = dict(BASE, **LOSSES[args.loss])
    train, test = load_split("TRAIN"), load_split("TEST")
    sigma_eval = eval_sigmanet(train, args.device)
    out = Path(args.out or RESULTS / "ecg5000_{}.json".format(args.loss))
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        f, sigma, t = train_one(args.loss, cfg, seed, train, args.device)
        if args.save_ckpt:
            ckpt = Path(args.save_ckpt)
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save(dict(forecaster=f.state_dict(), sigmanet=sigma.state_dict(), config=cfg),
                       ckpt / "ecg5000_{}_seed{}.pt".format(args.loss, seed))
        runs.append(dict(seed=seed, train_time_s=t, **evaluate(f, sigma_eval, test, cfg)))
        print(json.dumps(runs[-1]), flush=True)
        out.write_text(json.dumps(dict(loss=args.loss, config=cfg, torch=torch.__version__,
                                       runs=runs), indent=2))


if __name__ == "__main__":
    main()
