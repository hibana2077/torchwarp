"""Path figures of soft-DTW, FVM and JEANIE on the NW-UCLA test split.

Train and save the JEANIE encoder first:
  python experiments/nwucla_fewshot.py --method jeanie --seeds 42 --save-ckpt experiments/results/ckpt

Every test query (classes 6, 8, 9, 11, 12, view 3) is compared with one
support sequence of the same action from views 1 and 2: by default a
random one (--support random, fixed --seed), or the one with the smallest
JEANIE distance (--support nearest). One figure is written per query, plus index.csv
with the distances shown in the figures:
  python experiments/nwucla_plot_paths.py

One pair only (the README figure is --query 970 --support 1005):
  python experiments/nwucla_plot_paths.py --query 970 --support 1005 --out assets
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torchwarp
from torchwarp import paths, plot

import nwucla_fewshot as nw

ROOT = Path(__file__).resolve().parents[1]

ACTIONS = {1: "pick up with one hand", 2: "pick up with two hands", 3: "drop trash",
           4: "walk around", 5: "sit down", 6: "stand up", 8: "donning", 9: "doffing",
           11: "throw", 12: "carry"}
BONES = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7), (2, 8), (8, 9), (9, 10),
         (10, 11), (0, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)]


class Encoded:
    """Blocks (stride 1), features and 2D poses of the NW-UCLA sequences."""

    def __init__(self, ckpt, device):
        ck = torch.load(ckpt, map_location="cpu")
        self.cfg = dict(ck["config"], stride=1)
        self.seqs = nw.load_dataset(self.cfg)
        self.rots = nw.rotations(self.cfg)
        self.enc = nw.BlockEncoder(self.cfg["block"] * 20 * 3, self.cfg).to(device).eval()
        self.enc.load_state_dict(ck["encoder"])
        self.device = device
        c = self.cfg
        self.mid = [t + c["block"] // 2 for t in range(0, c["frames"] - c["block"] + 1, 1)]

    def skeleton(self, i):
        return torch.tensor(self.seqs[i][2], dtype=torch.float32)

    @torch.no_grad()
    def query(self, i):
        return self.enc(nw.query_views(self.skeleton(i), self.rots, self.cfg).to(self.device))

    @torch.no_grad()
    def support(self, i):
        return self.enc(nw.blocks(self.skeleton(i), self.cfg).to(self.device))

    def query_poses(self, i):
        return torch.einsum("kab,fjb->kfja", self.rots, self.skeleton(i))[:, self.mid][..., :2]

    def support_poses(self, i):
        return self.skeleton(i)[self.mid][..., :2]


def nearest_support(data, q, candidates, gamma, max_shift):
    query = data.query(q)
    supports = torch.stack([data.support(s) for s in candidates])
    d = torchwarp.jeanie_1d_from_features(query.expand(len(candidates), -1, -1, -1), supports,
                                          gamma, max_shift)
    return candidates[int(d.argmin())]


def draw(data, q, s, path, gamma, shifts):
    query, support = data.query(q), data.support(s)
    fig = plot.viewpoint_figure(query, support, data.cfg["angles_deg"], gamma=gamma,
                                max_shift=shifts, query_poses=data.query_poses(q),
                                support_poses=data.support_poses(s), bones=BONES, pose_step=3)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    row = dict(query=q, support=s, action=ACTIONS[data.seqs[q][0]],
               query_view=data.seqs[q][1], support_view=data.seqs[s][1])
    per_view = paths.sdtw_per_view(query, support, gamma)
    row["d_sdtw_min"] = min(float(a.distance[0]) for a in per_view)
    row["d_fvm"] = float(paths.fvm(query, support, gamma).distance[0])
    for m in shifts:
        row["d_jeanie_shift{}".format(m)] = float(
            torchwarp.jeanie_1d_from_features(query, support, gamma, m))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "experiments/results/ckpt/nwucla_jeanie_seed42.pt"))
    ap.add_argument("--out", default=str(ROOT / "experiments/results/figures/nwucla"))
    ap.add_argument("--query", type=int, default=None, help="dataset index of one query")
    ap.add_argument("--support", default="random",
                    help="'random', 'nearest', or the dataset index of one support")
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--max-shift", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--limit", type=int, default=None, help="draw only the first N queries")
    ap.add_argument("--seed", type=int, default=0, help="for --support random")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data = Encoded(args.ckpt, args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, seqs = data.cfg, data.seqs
    shifts = tuple(args.max_shift)

    if args.query is not None and args.support not in ("nearest", "random"):
        row = draw(data, args.query, int(args.support), out / "viewpoint_paths_nwucla.png",
                   args.gamma, shifts)
        print(row)
        return

    queries = [args.query] if args.query is not None else [
        i for i, (label, view, _) in enumerate(seqs)
        if label in cfg["test_classes"] and view == 3]
    queries = queries[:args.limit]
    rng = np.random.RandomState(args.seed)
    rows = []
    for n, q in enumerate(queries):
        label = seqs[q][0]
        candidates = [i for i, (l, v, _) in enumerate(seqs) if l == label and v in (1, 2)]
        if args.support == "random":
            s = int(rng.choice(candidates))
        else:
            s = nearest_support(data, q, candidates, args.gamma, max(shifts))
        name = "{:04d}_{}_s{:04d}.png".format(q, ACTIONS[label].replace(" ", "-"), s)
        rows.append(draw(data, q, s, out / name, args.gamma, shifts))
        print("[{}/{}] {}".format(n + 1, len(queries), name), flush=True)
        with open(out / "index.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
