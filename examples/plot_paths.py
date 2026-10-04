"""Draw the two path figures of the README from trained example models.

Train and save the models first:
  python examples/ecg5000_forecast.py --loss udtw --seeds 42 --save-ckpt examples/results/ckpt
  python examples/nwucla_fewshot.py --method jeanie --seeds 42 --save-ckpt examples/results/ckpt

Then:
  python examples/plot_paths.py

uDTW: one ECG5000 test series (--ecg-index); the forecast of the trained
uDTW model and the ground truth are compared, with sigma from its SigmaNet.
JEANIE: a query and a support sequence of the same NW-UCLA action
(--query, --support: indices into the loaded dataset), encoded by the trained
JEANIE block encoder; the query is rotated to each simulated viewpoint.
"""

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")

import ecg5000_forecast as ecg
import nwucla_fewshot as nw
from torchwarp import plot

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "examples" / "results" / "ckpt"

BONES = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7), (2, 8), (8, 9), (9, 10),
         (10, 11), (0, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)]


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


def viewpoint_figure(path, query, support, device):
    ck = torch.load(CKPT / "nwucla_jeanie_seed42.pt", map_location="cpu")
    cfg = dict(ck["config"], stride=1)
    seqs = nw.load_dataset(cfg)
    rots = nw.rotations(cfg)
    enc = nw.BlockEncoder(cfg["block"] * 20 * 3, cfg).to(device)
    enc.load_state_dict(ck["encoder"])
    qseq = torch.tensor(seqs[query][2], dtype=torch.float32)
    sseq = torch.tensor(seqs[support][2], dtype=torch.float32)
    with torch.no_grad():
        q = enc(nw.query_views(qseq, rots, cfg).to(device))
        s = enc(nw.blocks(sseq, cfg).to(device))
    mid = [t + cfg["block"] // 2 for t in range(0, cfg["frames"] - cfg["block"] + 1, cfg["stride"])]
    q_poses = torch.einsum("kab,fjb->kfja", rots, qseq)[:, mid][..., :2]
    s_poses = sseq[mid][..., :2]
    fig = plot.viewpoint_figure(q, s, cfg["angles_deg"], gamma=0.1, max_shift=(1, 2),
                                query_poses=q_poses, support_poses=s_poses, bones=BONES,
                                pose_step=3)
    fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg-index", type=int, default=0)
    ap.add_argument("--query", type=int, default=970)
    ap.add_argument("--support", type=int, default=1005)
    ap.add_argument("--out", default=str(ROOT / "assets"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    udtw_figure(out / "udtw_paths_ecg5000.png", args.ecg_index, args.device)
    viewpoint_figure(out / "viewpoint_paths_nwucla.png", args.query, args.support, args.device)


if __name__ == "__main__":
    main()
