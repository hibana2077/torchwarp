"""ECG5000 forecasting (UCR ECG5000, 500 train / 4500 test series of length 140).

An MLP predicts the last 40% of each series (56 steps) from the first 60%
(84 steps). Training losses (squared-Euclidean frame cost):
  * euclidean: sum_t (pred_t - y_t)^2
  * dtw:       hard DTW (plain PyTorch dynamic program)
  * sdtw:      soft-DTW sdtw(x,y)
  * sdtw_div:  soft-DTW divergence sdtw(x,y) - [sdtw(x,x) + sdtw(y,y)] / 2
  * udtw:      d_uDTW + beta * Omega, sigma from a SigmaNet

Two setups, both with Sigma = 0.5 (sigma^2 + sigma'^2) (Eq. 8):
  * default: ReLU MLP 84 -> 256 -> 56 with PyTorch's default init, Adam,
    and a SigmaNet MLP over the whole sequence.
  * paper:   the training settings stated in the uDTW paper (arXiv
    2211.00005, App. H.2, H.5, I.2): tanh MLP 84 -> t' -> 56 with every
    weight drawn from N(0, 1), SGD (lr 1e-3, weight decay 1e-6), and a
    per-frame SigmaNet sigma_t = kappa * sigmoid(FC(psi_t)) + eta
    (kappa = 1.5, eta = 0.01; one value per frame, as in App. H.2 and the
    official SimpleSigmaNet). The paper
    leaves t', epochs, batch size, gamma and beta unspecified; PAPER_LOSSES
    holds values selected on a 400/100 validation split of the training set
    (experiments/ecg5000_tune.py --setup paper).

Test metrics (float64, lower is better): MSE (per time step), SSE (summed
over the 56 predicted steps), DTW, sDTW div. (gamma=1) and uDTW (gamma=1,
beta=1, normalize=True). The uDTW metric takes sigma from one fixed
evaluation SigmaNet (a uDTW run with seed EVAL_SEED), shared by every model.

Usage:
  python experiments/ecg5000_forecast.py --loss udtw
  python experiments/ecg5000_forecast.py --setup paper --loss udtw --seeds $(seq 0 99)
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchwarp import UDTW, soft_dtw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "data"
RESULTS = ROOT / "experiments" / "results"

BASE = dict(
    input_len=84, output_len=56, hidden=256, act="relu", init="default",
    sigma_hidden=64, sigma_a=1.5, sigma_b=0.5,                   # sigma = a * sigmoid(.) + b
    gamma=1.0, beta=1.0, normalize=True,
    optimizer="adam", lr=1e-3, weight_decay=0.0, batch_size=50, epochs=100,
)
LOSSES = {
    "euclidean": {},
    "dtw": {},
    "sdtw": dict(gamma=0.001),
    "sdtw_div": dict(gamma=0.001),
    "udtw": dict(gamma=0.1, beta=0.01, sigma_a=2.0, sigma_b=0.1, lr=1e-3,     # ecg5000_tune.py
                 normalize=False),
}
PAPER_BASE = dict(
    input_len=84, output_len=56, hidden=50, act="tanh", init="normal",
    sigmanet="frame", sigma_a=1.5, sigma_b=0.01,                 # kappa, eta (App. I.2)
    gamma=1.0, beta=1.0, normalize=False,
    optimizer="sgd", lr=1e-3, weight_decay=1e-6, batch_size=1, epochs=100,
)
PAPER_LOSSES = {                                    # ecg5000_tune.py --setup paper select
    "euclidean": dict(hidden=30, batch_size=1, epochs=50),
    "dtw": dict(hidden=30, batch_size=10, epochs=100),
    "sdtw": dict(hidden=30, batch_size=1, epochs=50, gamma=0.1),
    "sdtw_div": dict(hidden=30, batch_size=1, epochs=100, gamma=0.1),
    "udtw": dict(hidden=50, batch_size=1, epochs=100, gamma=0.01, beta=0.1),
}
SETUPS = {"default": (BASE, LOSSES), "paper": (PAPER_BASE, PAPER_LOSSES)}
EVAL_SEED = 100


def config(setup, loss, overrides=()):
    base, losses = SETUPS[setup]
    cfg = dict(base, **losses[loss])
    for kv in overrides:
        key, value = kv.split("=", 1)
        cfg[key] = type(base[key])(value) if not isinstance(base[key], bool) else value == "True"
    return cfg


def load_split(name):
    arr = np.loadtxt(DATA / "ECG5000_{}.txt".format(name))
    return torch.tensor(arr[:, 1:], dtype=torch.float32)  # drop the class label


def split_validation(train, n_val=100):
    """Fixed 400/100 train/validation split of the training set."""
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))
    return train[perm[n_val:]], train[perm[:n_val]]


class Forecaster(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        act = nn.Tanh() if cfg.get("act", "relu") == "tanh" else nn.ReLU()
        self.net = nn.Sequential(nn.Linear(cfg["input_len"], cfg["hidden"]), act,
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


class FrameSigmaNet(nn.Module):
    """Per-frame SigmaNet (App. H.2, Eq. 25): sigma_t = kappa * sigmoid(FC(psi_t)) + eta,
    FC from the frame feature (1-D here) to a scalar."""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(1, 1)
        self.a, self.b = cfg["sigma_a"], cfg["sigma_b"]

    def forward(self, z):
        return self.a * torch.sigmoid(self.fc(z.unsqueeze(-1)).squeeze(-1)) + self.b


def make_models(cfg):
    f = Forecaster(cfg)
    sigma = FrameSigmaNet(cfg) if cfg.get("sigmanet") == "frame" else SigmaNet(cfg)
    if cfg.get("init") == "normal":                      # App. H.5: weights ~ N(0, 1)
        for p in list(f.parameters()) + list(sigma.parameters()):
            nn.init.normal_(p, 0.0, 1.0)
    return f, sigma


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
    crit = UDTW(gamma=cfg["gamma"], normalize=cfg["normalize"])

    def loss(pred, y, sigma):
        d, p = crit(pred.unsqueeze(-1), y.unsqueeze(-1),
                    sigma(pred).unsqueeze(-1), sigma(y).unsqueeze(-1), beta=cfg["beta"])
        return (d + p).mean()
    return loss


def train_one(name, cfg, seed, train, device, on_epoch=None):
    """Train one model; on_epoch(epoch, f, sigma) is called after each epoch."""
    torch.manual_seed(seed)
    f, sigma = make_models(cfg)
    f, sigma = f.to(device), sigma.to(device)
    crit = make_loss(name, cfg)
    params = list(f.parameters()) + (list(sigma.parameters()) if name == "udtw" else [])
    if cfg.get("optimizer", "adam") == "sgd":
        opt = torch.optim.SGD(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        opt = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    x_all, y_all = train[:, :cfg["input_len"]], train[:, cfg["input_len"]:]
    gen = torch.Generator().manual_seed(seed)

    step_time = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
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
        if on_epoch is not None:
            on_epoch(epoch, f, sigma)
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


def eval_sigmanet(setup, train, device):
    """The fixed SigmaNet of the uDTW test metric (trained once, then cached)."""
    cfg = BASE if setup == "default" else config(setup, "udtw")
    suffix = "" if setup == "default" else "_" + setup
    path = RESULTS / "ecg5000_eval_sigmanet{}_seed{}.pt".format(suffix, EVAL_SEED)
    _, sigma = make_models(cfg)
    if path.exists():
        sigma.load_state_dict(torch.load(path))
    else:
        _, sigma, _ = train_one("udtw", cfg, EVAL_SEED, train, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sigma.state_dict(), path)
    return sigma.double().eval()


def evaluate(f, sigma_eval, test, cfg):
    """Test metrics; sigma_eval = the evaluation SigmaNet, or None to skip uDTW."""
    metric = UDTW(gamma=1.0, normalize=True, backend="torch")
    x = test[:, :cfg["input_len"]].double()
    y = test[:, cfg["input_len"]:].double()
    with torch.no_grad():
        pred = f.double()(x)
        p3, y3 = pred.unsqueeze(-1), y.unsqueeze(-1)
        one = torch.ones_like(y3)
        sdtw_div, _ = metric(p3, y3, one, one, beta=0.0)      # sigma = 1: soft-DTW
        se = (pred - y) ** 2
        out = dict(
            mse=float(se.mean()),
            sse=float(se.sum(1).mean()),
            dtw=float(hard_dtw(pred, y).mean()),
            sdtw_div=float(sdtw_div.mean()),
        )
        if sigma_eval is not None:
            d, pen = metric(p3, y3, sigma_eval(pred).unsqueeze(-1), sigma_eval(y).unsqueeze(-1),
                            beta=1.0)
            out["udtw"] = float((d + pen).mean())
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=sorted(LOSSES), required=True)
    ap.add_argument("--setup", choices=sorted(SETUPS), default="default")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val", action="store_true",
                    help="train on 400 training series, evaluate on the other 100 (no uDTW metric)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config entries, e.g. --set epochs=30 gamma=0.1")
    ap.add_argument("--eval-epochs", type=int, nargs="*", default=[],
                    help="also evaluate after these epochs (stored under 'at_epoch')")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-ckpt", default=None, metavar="DIR")
    args = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = config(args.setup, args.loss, args.set)
    train, test = load_split("TRAIN"), None
    if args.val:
        train, test = split_validation(train)
        sigma_eval = None
    else:
        test = load_split("TEST")
        sigma_eval = eval_sigmanet(args.setup, train, args.device)
    suffix = "" if args.setup == "default" else "_" + args.setup
    out = Path(args.out or RESULTS / "ecg5000{}_{}.json".format(suffix, args.loss))
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        at_epoch = {}

        def on_epoch(epoch, f, sigma):
            if epoch in args.eval_epochs and epoch != cfg["epochs"]:
                at_epoch[epoch] = evaluate(copy.deepcopy(f).cpu(), sigma_eval, test, cfg)

        f, sigma, t = train_one(args.loss, cfg, seed, train, args.device, on_epoch)
        if args.save_ckpt:
            ckpt = Path(args.save_ckpt)
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save(dict(forecaster=f.state_dict(), sigmanet=sigma.state_dict(), config=cfg),
                       ckpt / "ecg5000{}_{}_seed{}.pt".format(suffix, args.loss, seed))
        runs.append(dict(seed=seed, train_time_s=t, **evaluate(f, sigma_eval, test, cfg)))
        if at_epoch:
            runs[-1]["at_epoch"] = at_epoch
        print(json.dumps(runs[-1]), flush=True)
        out.write_text(json.dumps(dict(loss=args.loss, setup=args.setup, val=args.val, config=cfg,
                                       torch=torch.__version__, runs=runs), indent=2))


if __name__ == "__main__":
    main()
