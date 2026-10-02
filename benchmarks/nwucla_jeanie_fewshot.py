"""Cross-view few-shot skeleton action recognition with JEANIE (NW-UCLA).

The JEANIE paper evaluates on NTU-60/120, Kinetics-skeleton and UWA3D
Multiview II. NTU needs a registration, the UWA3D site is offline and
Kinetics-skeleton has no 3D/multi-view data. This benchmark therefore uses
the public NW-UCLA Multiview 3D skeleton dataset (3 Kinect views, 10
actions, 20 joints) in a protocol that mirrors the paper's FSAR pipeline:

  * temporal blocks: 32 resampled frames, blocks of M=8 frames, stride 4
    (T=7 blocks);
  * viewpoint simulation of the query: rotations about the vertical axis by
    {-30,-15,0,15,30} degrees (K=5);
  * block encoder: MLP 480 -> 256 -> 64;
  * distance: JEANIE-1D (gamma=0.1, iota=1, Euclidean base distance);
  * training: 5-way 1-shot episodes on 5 training classes (views 1+2),
    cross-entropy over -JEANIE distances;
  * testing: 5-way 1-shot episodes on the 5 held-out classes, supports from
    views 1+2 and queries from view 3 (cross-view).

Reference (github.com/LeiWangR/JEANIE, one pair per call) and torchwarp
(batched) runs share data, episodes, initial weights and hyperparameters.

Usage:
  python benchmarks/nwucla_jeanie_fewshot.py --impl ref  --device cpu  --seeds 0 1 2 3 4
  python benchmarks/nwucla_jeanie_fewshot.py --impl fast --device cuda --seeds 0 1 2 3 4
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

from torchwarp.testing import load_reference

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "data" / "nwucla" / "all_sqe"

CONFIG = dict(
    dataset="NW-UCLA Multiview (all_sqe), 20 joints, 3 views",
    frames=32, block=8, stride=4, angles_deg=[-30, -15, 0, 15, 30],
    train_classes=[1, 2, 3, 4, 5], test_classes=[6, 8, 9, 11, 12],
    hidden=256, embed=64,
    gamma=0.1, max_shift=1, temperature=1.0,
    way=5, shot=1, train_queries_per_class=1, test_queries_per_class=5,
    train_episodes=300, test_episodes=300,
    optimizer="Adam", lr=1e-3, dtype="float32",
)


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


def make_distance(impl, cfg):
    """All-pairs JEANIE distances: q [nq,K,T,E], s [ns,T,E] -> [nq, ns]."""
    if impl == "ref":
        ref = load_reference("jeanie")

        def dist(q, s):
            rows = [torch.stack([ref.jeanie_1d_from_cost(ref.euclidean_cost(qi, sj),
                                                         cfg["gamma"], cfg["max_shift"])
                                 for sj in s]) for qi in q]
            return torch.stack(rows)
        return dist

    import torchwarp as fast

    def dist(q, s):
        nq, ns = q.shape[0], s.shape[0]
        qq = q.unsqueeze(1).expand(nq, ns, *q.shape[1:]).reshape(nq * ns, *q.shape[1:])
        ss = s.unsqueeze(0).expand(nq, ns, *s.shape[1:]).reshape(nq * ns, *s.shape[1:])
        d = fast.jeanie_1d_from_features(qq, ss, cfg["gamma"], cfg["max_shift"])
        return d.view(nq, ns)
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


def run(impl, device, seed, cfg, seqs):
    dtype = getattr(torch, cfg["dtype"])
    rots = rotations(cfg).to(dtype)
    support_blocks = [blocks(torch.tensor(s, dtype=dtype), cfg) for _, _, s in seqs]
    query_blocks = [query_views(torch.tensor(s, dtype=dtype), rots, cfg)
                    for _, _, s in seqs]
    pool = {}
    for idx, (label, view, _) in enumerate(seqs):
        pool.setdefault(label, []).append((idx, view))

    torch.manual_seed(seed)
    enc = BlockEncoder(support_blocks[0].shape[1], cfg).to(device, dtype)
    opt = torch.optim.Adam(enc.parameters(), lr=cfg["lr"])
    dist = make_distance(impl, cfg)
    rng = np.random.RandomState(seed)

    def logits(sup, qry):
        s = enc(torch.stack([support_blocks[i] for i in sup]).to(device))      # [ns,T,E]
        q = enc(torch.stack([query_blocks[i] for i in qry]).to(device))        # [nq,K,T,E]
        return -dist(q, s) / cfg["temperature"]

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    train_time, losses = 0.0, []
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
        losses.append(loss.item())

    test_rng = np.random.RandomState(10_000 + seed)
    correct, total, test_time = 0, 0, 0.0
    with torch.no_grad():
        for _ in range(cfg["test_episodes"]):
            sup, qry, lab = sample_episode(test_rng, pool, cfg["test_classes"], cfg,
                                           cfg["test_queries_per_class"], (3,), (1, 2))
            sync()
            t0 = time.perf_counter()
            pred = logits(sup, qry).argmax(1).cpu()
            sync()
            test_time += time.perf_counter() - t0
            correct += int((pred == lab).sum())
            total += len(lab)
    return dict(seed=seed, accuracy=100.0 * correct / total,
                final_train_loss=float(np.mean(losses[-20:])),
                train_time_s=train_time, test_time_s=test_time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["ref", "fast"], required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=None)
    ap.add_argument("--episodes", type=int, default=None, help="smoke tests only")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = ap.parse_args()
    CONFIG["dtype"] = args.dtype
    if args.episodes is not None:
        CONFIG["train_episodes"] = CONFIG["test_episodes"] = args.episodes

    seqs = load_dataset(CONFIG)
    out = Path(args.out or ROOT / "benchmarks" / "results" /
               "nwucla_{}_{}.json".format(args.impl, args.device))
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        runs.append(run(args.impl, args.device, seed, CONFIG, seqs))
        print(json.dumps(runs[-1]), flush=True)
        out.write_text(json.dumps(dict(impl=args.impl, device=args.device, config=CONFIG,
                                       n_sequences=len(seqs), torch=torch.__version__,
                                       runs=runs), indent=2))


if __name__ == "__main__":
    main()
