"""Forward+backward timing: reference uDTW vs torchwarp backends.

    python benchmarks/speed_udtw.py
"""

import time

import torch

from torchwarp import _cuda, uDTW
from torchwarp.testing import load_reference

RefUDTW = load_reference("udtw").uDTW


def _step(module, inputs):
    xs = [x.detach().requires_grad_(True) for x in inputs]
    dist, pen = module(*xs, beta=1.0)
    torch.autograd.grad(dist.sum() + pen.sum(), xs)


def timeit(module, inputs, reps):
    cuda = inputs[0].is_cuda
    _step(module, inputs)
    if cuda:
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        _step(module, inputs)
        if cuda:
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def make(B, L, D=64, device="cpu"):
    g = torch.Generator().manual_seed(0)
    xs = (
        torch.rand(B, L, D, generator=g),
        torch.rand(B, L, D, generator=g),
        0.5 + torch.rand(B, L, 1, generator=g),
        0.5 + torch.rand(B, L, 1, generator=g),
    )
    return [x.to(device) for x in xs]


def main():
    has_cuda = _cuda.available("udtw")
    print(torch.__version__, torch.cuda.get_device_name(0) if has_cuda else "cpu only")
    header = "{:30s} {:>11s} {:>11s} {:>11s} {:>11s} {:>9s}".format(
        "case (fwd+bwd, fp32, ms)", "ref CPU", "torch CPU", "torch GPU", "CUDA", "vs ref")
    print(header)
    print("-" * len(header))
    cases = [
        (8, 10, False), (8, 25, False), (8, 50, False), (8, 25, True),
        (64, 25, False), (256, 25, False), (1024, 25, False), (256, 50, True),
    ]
    for B, L, norm in cases:
        name = "B{} N=M={}{}".format(B, L, " norm" if norm else "")
        cpu = make(B, L)
        row = []
        if B <= 64:
            row.append(timeit(RefUDTW(gamma=0.1, normalize=norm), cpu, 2))
        else:
            row.append(float("nan"))
        row.append(timeit(uDTW(gamma=0.1, normalize=norm, backend="torch"), cpu, 3))
        if has_cuda:
            gpu = make(B, L, device="cuda")
            row.append(timeit(uDTW(gamma=0.1, normalize=norm, backend="torch"), gpu, 5))
            row.append(timeit(uDTW(gamma=0.1, normalize=norm, backend="cuda"), gpu, 20))
        else:
            row += [float("nan")] * 2
        speedup = row[0] / row[3] if row[0] == row[0] and row[3] == row[3] else float("nan")
        print("{:30s} {:11.2f} {:11.2f} {:11.2f} {:11.3f} {:8.0f}x".format(name, *row, speedup))


if __name__ == "__main__":
    main()
