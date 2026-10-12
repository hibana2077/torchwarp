"""soft-DTW and uDTW path figure on ECG5000 (assets/udtw_paths_ecg5000.png).

Train and save the uDTW model first:
  python experiments/ecg5000_forecast.py --loss udtw --seeds 42 --save-ckpt experiments/results/ckpt

Then:
  python experiments/ecg5000_plot_paths.py

One ECG5000 test series (--index): the forecast of the trained uDTW model
and the ground truth are compared, with sigma from its SigmaNet. The NW-UCLA
figure is drawn by nwucla_plot_paths.py.
"""

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")

from torchwarp import plot

import ecg5000_forecast as ecg

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "experiments" / "results" / "ckpt"


def udtw_figure(path, index, device):
    ck = torch.load(CKPT / "ecg5000_udtw_seed42.pt", map_location="cpu")
    cfg = ck["config"]
    f, sigma = ecg.Forecaster(cfg), ecg.SigmaNet(cfg)
    f.load_state_dict(ck["forecaster"])
    sigma.load_state_dict(ck["sigmanet"])
    test = ecg.load_split("TEST")
    with torch.no_grad():
        x = test[index:index + 1, :cfg["input_len"]]
        y = test[index:index + 1, cfg["input_len"]:]
        pred = f(x)
        X, Y = pred.unsqueeze(-1).to(device), y.unsqueeze(-1).to(device)
        sx, sy = sigma(pred).unsqueeze(-1).to(device), sigma(y).unsqueeze(-1).to(device)
    fig = plot.udtw_figure(X, Y, sx, sy, gammas=(0.01, 0.1), beta=1.0)
    fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "assets"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    udtw_figure(out / "udtw_paths_ecg5000.png", args.index, args.device)


if __name__ == "__main__":
    main()
