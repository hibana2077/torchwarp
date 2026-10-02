"""Forward+backward timing: reference JEANIE vs torchwarp backends.

The reference only takes one pair per call, so it is timed as a Python loop
over the batch.

    python benchmarks/speed_jeanie.py
"""

import time

import torch

import torchwarp as fast
from torchwarp import _cuda
from torchwarp.testing import load_reference

ref = load_reference("jeanie")


def timeit(fn, inputs, reps):
    cuda = inputs[0].is_cuda

    def step():
        xs = [x.detach().requires_grad_(True) for x in inputs]
        torch.autograd.grad(fn(*xs).sum(), xs)

    step()
    if cuda:
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        step()
        if cuda:
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def case_1d(B, K, L, D=64):
    g = torch.Generator().manual_seed(0)
    q, s = torch.randn(B, K, L, D, generator=g), torch.randn(B, L, D, generator=g)

    def fast_fn(backend):
        return lambda q, s: fast.jeanie_1d_from_cost(fast.euclidean_cost(q, s), 0.1, 1, backend=backend)

    fast_fn.features = lambda q, s: fast.jeanie_1d_from_features(q, s, 0.1, 1, backend="cuda")

    def ref_fn(q, s):
        return torch.stack([ref.jeanie_1d_from_cost(ref.euclidean_cost(a, b), 0.1, 1)
                            for a, b in zip(q, s)])
    return [q, s], ref_fn, fast_fn


def case_2d(B, K1, K2, L, D=64):
    g = torch.Generator().manual_seed(0)
    q, s = torch.randn(B, K1 * K2, L, D, generator=g), torch.randn(B, L, D, generator=g)

    def fast_fn(backend):
        return lambda q, s: fast.jeanie_2d_from_cost(
            fast.euclidean_cost(q, s).reshape(B, K1, K2, L, L), 0.1, 1, 1, backend=backend)

    fast_fn.features = lambda q, s: fast.jeanie_2d_from_features(
        q.reshape(B, K1, K2, L, -1), s, 0.1, 1, 1, backend="cuda")

    def ref_fn(q, s):
        return torch.stack([ref.jeanie_2d_from_cost(
            ref.euclidean_cost(a, b).reshape(K1, K2, L, L), 0.1, 1, 1) for a, b in zip(q, s)])
    return [q, s], ref_fn, fast_fn


def case_fvm(B, K, L, D=64):
    g = torch.Generator().manual_seed(0)
    q, s = torch.randn(B, K, L, D, generator=g), torch.randn(B, L, D, generator=g)

    def fast_fn(backend):
        return lambda q, s: fast.fvm_query_only_1d(fast.euclidean_cost(q, s), 0.1, backend=backend)

    def ref_fn(q, s):
        return torch.stack([ref.fvm_query_only_1d(ref.euclidean_cost(a, b), 0.1)
                            for a, b in zip(q, s)])
    return [q, s], ref_fn, fast_fn


def main():
    has_cuda = _cuda.available("jeanie")
    print(torch.__version__, torch.cuda.get_device_name(0) if has_cuda else "cpu only")
    header = "{:32s} {:>10s} {:>10s} {:>10s} {:>10s} {:>11s} {:>9s}".format(
        "case (fwd+bwd, fp32, ms)", "ref CPU", "torch CPU", "torch GPU", "CUDA cost",
        "CUDA fused", "vs ref")
    print(header)
    print("-" * len(header))
    cases = [
        ("JEANIE-1D B1 K5 T=U=10", case_1d(1, 5, 10)),
        ("JEANIE-1D B1 K5 T=U=30", case_1d(1, 5, 30)),
        ("JEANIE-1D B8 K5 T=U=20", case_1d(8, 5, 20)),
        ("JEANIE-1D B256 K5 T=U=20", case_1d(256, 5, 20)),
        ("JEANIE-2D B1 3x3 T=U=15", case_2d(1, 3, 3, 15)),
        ("JEANIE-2D B8 5x5 T=U=20", case_2d(8, 5, 5, 20)),
        ("JEANIE-2D B256 5x5 T=U=20", case_2d(256, 5, 5, 20)),
        ("FVM B8 K5 T=U=30", case_fvm(8, 5, 30)),
        ("FVM B256 K5 T=U=30", case_fvm(256, 5, 30)),
    ]
    for name, (inputs, ref_fn, fast_fn) in cases:
        B = inputs[0].shape[0]
        cells = inputs[0].shape[1] * inputs[0].shape[2] ** 2
        row = [timeit(ref_fn, inputs, 1) if B * cells <= 8 * 25 * 400 else float("nan")]
        row.append(timeit(fast_fn("torch"), inputs, 3))
        if has_cuda:
            gpu = [x.cuda() for x in inputs]
            row.append(timeit(fast_fn("torch"), gpu, 5))
            row.append(timeit(fast_fn("cuda"), gpu, 20))
            features = getattr(fast_fn, "features", None)
            row.append(timeit(features, gpu, 20) if features else float("nan"))
        else:
            row += [float("nan")] * 3
        best = min(v for v in row[3:] if v == v) if has_cuda else float("nan")
        speedup = row[0] / best if row[0] == row[0] and best == best else float("nan")
        print("{:32s} {:10.2f} {:10.2f} {:10.2f} {:10.3f} {:11.3f} {:8.0f}x".format(
            name, *row, speedup))


if __name__ == "__main__":
    main()
