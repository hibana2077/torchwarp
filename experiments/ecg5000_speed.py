"""Training-step time of the ECG5000 uDTW loss: official uDTW vs torchwarp.

Same model, data and settings as `ecg5000_forecast.py --loss udtw` (batch
of 50 series); one step is forward pass, loss, backward pass and optimizer
step. Mean over 30 steps after 3 warm-up steps (which also trigger the
one-time JIT compilation of the CUDA extension); the GPU is synchronized
before and after each step.

Usage:
  python experiments/ecg5000_speed.py [--threads 10]
"""

import argparse
import time

import torch

from torchwarp import UDTW
from torchwarp._reference import load_reference

import ecg5000_forecast as E

WARMUP, STEPS = 3, 30


def official_criterion(cfg):
    """The official uDTW module; it takes beta in forward."""
    module = load_reference("udtw").uDTW(gamma=cfg["gamma"], normalize=cfg["normalize"])
    return lambda *xs: module(*xs, beta=cfg["beta"])


def torchwarp_criterion(cfg):
    return UDTW(gamma=cfg["gamma"], beta=cfg["beta"], normalize=cfg["normalize"])


def step_time(make_criterion, cfg, train, device):
    torch.manual_seed(42)
    f, sigma = E.make_models(cfg)
    f, sigma = f.to(device), sigma.to(device)
    crit = make_criterion(cfg)
    opt = torch.optim.Adam(list(f.parameters()) + list(sigma.parameters()), lr=cfg["lr"])
    x_all, y_all = train[:, :cfg["input_len"]].to(device), train[:, cfg["input_len"]:].to(device)
    times = []
    for k in range(WARMUP + STEPS):
        idx = torch.arange(k * cfg["batch_size"], (k + 1) * cfg["batch_size"]) % len(train)
        x, y = x_all[idx], y_all[idx]
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = f(x)
        d, p = crit(pred.unsqueeze(-1), y.unsqueeze(-1),
                    sigma(pred).unsqueeze(-1), sigma(y).unsqueeze(-1))
        loss = (d + p).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times[WARMUP:]) / STEPS * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = E.config("default", "udtw")
    train = E.load_split("TRAIN")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print("uDTW (ECG5000), ms per training step, batch {}".format(cfg["batch_size"]))
    for device in devices:
        ref = step_time(official_criterion, cfg, train, device)
        tw = step_time(torchwarp_criterion, cfg, train, device)
        print("{:4s}  official {:9.2f}  torchwarp {:8.2f}  speed-up {:6.1f}x".format(
            device, ref, tw, ref / tw), flush=True)


if __name__ == "__main__":
    main()
