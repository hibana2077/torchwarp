"""Cross-view few-shot skeleton action recognition on NW-UCLA.

NW-UCLA Multiview 3D skeletons (10 actions, 20 joints, 3 Kinect views):
  * 32 resampled frames, temporal blocks of 8 frames with stride 4 (T=7);
  * query viewpoints simulated by rotations about the vertical axis;
  * block encoder: MLP (8*20*3 = 480) -> 256 -> 64;
  * training: 300 5-way 1-shot episodes on classes {1,2,3,4,5} (views 1+2,
    one query per class), cross-entropy over -distance / temperature;
  * testing: 300 5-way 1-shot episodes on classes {6,8,9,11,12}, supports
    from views 1+2, five queries per class from view 3.

Distances (Euclidean base distance between block embeddings):
  * sdtw, sdtw_div: soft-DTW / soft-DTW divergence on the 0-degree query view
  * udtw:           d_uDTW + beta * Omega (normalize=True) on the 0-degree view;
                    a head (64 -> 32 -> 1, sigma = 1.5 sigmoid + 0.5) predicts
                    a sigma per block
  * fvm:            query-only 1-D Free Viewpoint Matching over the K views
  * jeanie:         JEANIE-1D over the K views

Usage:
  python examples/nwucla_fewshot.py --method jeanie
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchwarp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data" / "nwucla" / "all_sqe"

BASE = dict(
    frames=32, block=8, stride=4, angles_deg=[-30, -15, 0, 15, 30],
    train_classes=[1, 2, 3, 4, 5], test_classes=[6, 8, 9, 11, 12],
    hidden=256, embed=64, sigma_hidden=32, sigma_a=1.5, sigma_b=0.5,
    way=5, shot=1, train_queries_per_class=1, test_queries_per_class=5,
    train_episodes=300, test_episodes=300,
    gamma=0.1, max_shift=1, beta=1.0, temperature=1.0, lr=1e-3,
)
METHODS = {
    "sdtw": dict(gamma=0.001),
    "sdtw_div": dict(gamma=0.001),
    "udtw": dict(gamma=0.1, beta=3.0, temperature=1.0, lr=3e-4),
    "fvm": dict(angles_deg=[-60, -30, 0, 30, 60], gamma=0.1, temperature=3.0, lr=3e-4),
    "jeanie": dict(angles_deg=[-60, -30, 0, 30, 60], gamma=1.0, max_shift=2,
                   temperature=10.0, lr=1e-3),
}


# ------------------------------------------------------------------- data

def load_dataset(cfg):
    seqs = []
    for path in sorted(DATA.glob("*.json")):
        meta = path.stem.split("_")                  # a01_s01_e00_v01
        label, view = int(meta[0][1:]), int(meta[3][1:])
        sk = np.asarray(json.loads(path.read_text())["skeletons"], dtype=np.float64)
        if sk.ndim != 3 or sk.shape[0] < 2:
            continue
        sk = sk - sk[0, 0]                           # centre on the first-frame hip
        t_old = np.linspace(0, 1, sk.shape[0])
        t_new = np.linspace(0, 1, cfg["frames"])
        flat = sk.reshape(sk.shape[0], -1)
        res = np.stack([np.interp(t_new, t_old, flat[:, k]) for k in range(flat.shape[1])], 1)
        seqs.append((label, view, res.reshape(cfg["frames"], -1, 3)))
    return seqs


def rotations(cfg):
    mats = []
    for deg in cfg["angles_deg"]:
        a = math.radians(deg)
        mats.append([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    return torch.tensor(mats, dtype=torch.float32)  # [K,3,3], rotation about y (up)


def blocks(seq, cfg):
    """[frames, J, 3] -> [T, block*J*3]."""
    starts = range(0, cfg["frames"] - cfg["block"] + 1, cfg["stride"])
    return torch.stack([seq[s:s + cfg["block"]].reshape(-1) for s in starts])


def query_views(seq, rots, cfg):
    """[frames, J, 3] -> [K, T, D] viewpoint-simulated blocks."""
    rotated = torch.einsum("kab,fjb->kfja", rots, seq)
    return torch.stack([blocks(r, cfg) for r in rotated])


# ------------------------------------------------------------------ model

class BlockEncoder(nn.Module):
    def __init__(self, in_dim, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, cfg["hidden"]), nn.ReLU(),
                                 nn.Linear(cfg["hidden"], cfg["embed"]))

    def forward(self, x):
        return self.net(x)


class SigmaHead(nn.Module):
    """Per-block sigma in [b, a + b] for uDTW."""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg["embed"], cfg["sigma_hidden"]), nn.ReLU(),
                                 nn.Linear(cfg["sigma_hidden"], 1))
        self.a, self.b = cfg["sigma_a"], cfg["sigma_b"]

    def forward(self, z):
        return self.a * torch.sigmoid(self.net(z)) + self.b


def all_pairs(q, s):
    """q [nq,...], s [ns,...] -> both [nq*ns,...], query-major."""
    nq, ns = q.shape[0], s.shape[0]
    qq = q.unsqueeze(1).expand(nq, ns, *q.shape[1:]).reshape(nq * ns, *q.shape[1:])
    ss = s.unsqueeze(0).expand(nq, ns, *s.shape[1:]).reshape(nq * ns, *s.shape[1:])
    return qq, ss


def make_distance(method, cfg, sigma_head):
    """All-pairs distances: q [nq,K,T,E], s [ns,T,E] -> [nq, ns]."""
    view0 = cfg["angles_deg"].index(0)
    gamma = cfg["gamma"]

    def sdtw(a, b):                                   # [P,T,E], [P,U,E] -> [P]
        return torchwarp.soft_dtw(torchwarp.euclidean_cost(a.unsqueeze(1), b)[:, 0], gamma)

    def udtw(a, b):
        d, pen = torchwarp.udtw_from_features(a, b, sigma_head(a), sigma_head(b),
                                              beta=cfg["beta"], gamma=gamma)
        return d + pen

    def dist(q, s):
        nq, ns = q.shape[0], s.shape[0]
        if method == "jeanie":
            qq, ss = all_pairs(q, s)
            return torchwarp.jeanie_1d_from_features(qq, ss, gamma, cfg["max_shift"]).view(nq, ns)
        if method == "fvm":
            qq, ss = all_pairs(q, s)
            return torchwarp.fvm_query_only_1d(torchwarp.euclidean_cost(qq, ss), gamma).view(nq, ns)
        q = q[:, view0]
        qq, ss = all_pairs(q, s)
        if method == "sdtw":
            return sdtw(qq, ss).view(nq, ns)
        fn = sdtw if method == "sdtw_div" else udtw
        return fn(qq, ss).view(nq, ns) - 0.5 * (fn(q, q)[:, None] + fn(s, s)[None, :])
    return dist


# --------------------------------------------------------------- episodes

def sample_episode(rng, pool, classes, cfg, n_query, query_views_allowed, support_views):
    chosen = rng.choice(classes, size=cfg["way"], replace=False)
    sup, qry, qlab = [], [], []
    for c_idx, c in enumerate(chosen):
        s_pool = [i for i in pool[c] if i[1] in support_views]
        q_pool = [i for i in pool[c] if i[1] in query_views_allowed]
        s_pick = rng.choice(len(s_pool), size=cfg["shot"], replace=False)
        support_ids = {s_pool[k][0] for k in s_pick}
        q_cands = [i for i in q_pool if i[0] not in support_ids]
        q_pick = rng.choice(len(q_cands), size=n_query, replace=False)
        sup += [s_pool[k][0] for k in s_pick]
        qry += [q_cands[k][0] for k in q_pick]
        qlab += [c_idx] * n_query
    return sup, qry, torch.tensor(qlab)


def run(method, cfg, seed, seqs, device):
    rots = rotations(cfg)
    support_blocks = [blocks(torch.tensor(s, dtype=torch.float32), cfg) for _, _, s in seqs]
    query_blocks = [query_views(torch.tensor(s, dtype=torch.float32), rots, cfg)
                    for _, _, s in seqs]
    pool = {}
    for idx, (label, view, _) in enumerate(seqs):
        pool.setdefault(label, []).append((idx, view))

    torch.manual_seed(seed)
    enc = BlockEncoder(support_blocks[0].shape[1], cfg).to(device)
    params = list(enc.parameters())
    sigma_head = None
    if method == "udtw":
        sigma_head = SigmaHead(cfg).to(device)
        params += list(sigma_head.parameters())
    dist = make_distance(method, cfg, sigma_head)
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    rng = np.random.RandomState(seed)

    def logits(sup, qry):
        s = enc(torch.stack([support_blocks[i] for i in sup]).to(device))      # [ns,T,E]
        q = enc(torch.stack([query_blocks[i] for i in qry]).to(device))        # [nq,K,T,E]
        return -dist(q, s) / cfg["temperature"]

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    train_time = 0.0
    for _ in range(cfg["train_episodes"]):
        sup, qry, lab = sample_episode(rng, pool, cfg["train_classes"], cfg,
                                       cfg["train_queries_per_class"], (1, 2), (1, 2))
        sync()
        t0 = time.perf_counter()
        loss = F.cross_entropy(logits(sup, qry), lab.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync()
        train_time += time.perf_counter() - t0

    test_rng = np.random.RandomState(10_000 + seed)
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(cfg["test_episodes"]):
            sup, qry, lab = sample_episode(test_rng, pool, cfg["test_classes"], cfg,
                                           cfg["test_queries_per_class"], (3,), (1, 2))
            pred = logits(sup, qry).argmax(1).cpu()
            correct += int((pred == lab).sum())
            total += len(lab)
    return dict(seed=seed, accuracy=100.0 * correct / total, train_time_s=train_time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=sorted(METHODS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = dict(BASE, **METHODS[args.method])
    seqs = load_dataset(cfg)
    out = Path(args.out or ROOT / "examples" / "results" / "nwucla_{}.json".format(args.method))
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        runs.append(run(args.method, cfg, seed, seqs, args.device))
        print(json.dumps(runs[-1]), flush=True)
        out.write_text(json.dumps(dict(method=args.method, config=cfg, torch=torch.__version__,
                                       runs=runs), indent=2))


if __name__ == "__main__":
    main()
