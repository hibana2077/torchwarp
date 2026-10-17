"""Write tests/golden/reference.pt: inputs and official outputs for every test case.

Run locally when a case is added or changed (downloads the official uDTW
and JEANIE code once, see torchwarp._reference):

    python tests/golden/generate.py

Everything runs on the CPU in float64.
"""

import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torchwarp._reference import load_reference  # noqa: E402

import golden as G  # noqa: E402

U = load_reference("udtw")
J = load_reference("jeanie")
f64 = dict(dtype=torch.float64)
OUT = {}


def case(name, /, **params):
    """Register a case; returns a generator seeded from its key."""
    k = G.key(name, **params)
    assert k not in OUT, k
    OUT[k] = {}
    return OUT[k], torch.Generator().manual_seed(zlib.crc32(k.encode()))


def detach(xs):
    return [x.detach().clone() for x in xs]


# ---------------------------------------------------------------- uDTW

def udtw_inputs(g, B, N, M, D=5):
    return [torch.rand(B, N, D, generator=g, **f64), torch.rand(B, M, D, generator=g, **f64),
            0.5 + torch.rand(B, N, 1, generator=g, **f64),
            0.5 + torch.rand(B, M, 1, generator=g, **f64)]


def udtw_run(inputs, beta=0.7, **kw):
    """Distance, penalty and the input gradients of sum(d) + 2 sum(penalty)."""
    xs = [x.clone().requires_grad_(True) for x in inputs]
    d, p = U.uDTW(**kw)(*xs, beta=beta)
    (d.sum() + 2.0 * p.sum()).backward()
    return detach([d, p]) + [x.grad for x in xs]


def udtw_scalar(beta=0.8, **kw):
    m = U.uDTW(**kw)

    def f(X, Y, SX, SY):
        d, p = m(X, Y, SX, SY, beta=beta)
        return d.sum() + 2.0 * p.sum()
    return f


for N, M in G.UDTW_SHAPES:
    for gamma in G.GAMMAS:
        for normalize in (False, True):
            c, g = case("udtw.matches", N=N, M=M, gamma=gamma, normalize=normalize)
            c["inputs"] = udtw_inputs(g, 3, N, M)
            c["outputs"] = udtw_run(c["inputs"], gamma=gamma, normalize=normalize)

for N, M, bandwidth in G.UDTW_BANDS:
    c, g = case("udtw.bandwidth", N=N, M=M, bandwidth=bandwidth)
    c["inputs"] = udtw_inputs(g, 2, N, M)
    c["outputs"] = udtw_run(c["inputs"], gamma=0.1, bandwidth=bandwidth)

for beta in G.UDTW_BETAS:
    c, g = case("udtw.beta", beta=beta)
    c["inputs"] = udtw_inputs(g, 2, 4, 6)
    c["outputs"] = udtw_run(c["inputs"], beta=beta, gamma=0.1)

c, g = case("udtw.pairwise_matrices")
c["inputs"] = udtw_inputs(g, 2, 4, 6)
c["outputs"] = detach(U.pairwise_matrices(*c["inputs"], beta=0.5))

c, g = case("udtw.softmin")
c["v"] = v = torch.randn(4, 5, generator=g, **f64)
c["outputs"] = [U.softmin(v, 0.3, dim=1), U.softmin(v, 0.3), U.softmin_weights(v, 0.3, dim=1)]

c, g = case("udtw.float32")
c["inputs"] = udtw_inputs(g, 16, 20, 25, D=32)
c["outputs"] = udtw_run(c["inputs"], gamma=0.1)

for normalize, bandwidth in G.UDTW_MAML:
    c, g = case("udtw.maml", normalize=normalize, bandwidth=bandwidth)
    c["inputs"] = X, Y, SX, SY = udtw_inputs(g, 2, 5, 6, D=4)
    c["W0"] = W0 = torch.randn(4, 4, generator=g, **f64)
    f = udtw_scalar(gamma=0.3, normalize=normalize, bandwidth=bandwidth)
    w = W0.clone().requires_grad_()
    gw, = torch.autograd.grad(f(X @ w.T, Y @ w.T, SX, SY), w, create_graph=True)
    w2 = w - 0.1 * gw
    c["meta_grad"] = torch.autograd.grad(f(X @ w2.T, Y @ w2.T, SX, SY), w)[0]

c, g = case("udtw.third_derivative")
c["inputs"] = X, Y, SX, SY = udtw_inputs(g, 2, 5, 6, D=4)
c["V"] = V = torch.randn(X.shape, generator=g, **f64)
t = torch.zeros((), **f64).requires_grad_()
y = udtw_scalar(gamma=0.5)(X + t * V, Y, SX, SY)
for _ in range(3):
    y, = torch.autograd.grad(y, t, create_graph=True)
c["d3"] = y.detach()

c, g = case("udtw.func")
c["inputs"] = X, Y, SX, SY = udtw_inputs(g, 1, 3, 4, D=2)
c["v"] = v = torch.randn(X.shape, generator=g, **f64)
loss = lambda x: udtw_scalar(gamma=0.4)(x, Y, SX, SY)  # noqa: E731
c["jacobian"] = torch.autograd.functional.jacobian(loss, X)
c["jvp"] = torch.autograd.functional.jvp(loss, X, v)[1]
c["hessian"] = torch.autograd.functional.hessian(loss, X)

c, g = case("udtw.vmap")
c["inputs"] = udtw_inputs(g, 2, 4, 5, D=4)
ds, ps = [], []
for scale in (1.0, 1.1, 0.9):
    d, p = U.uDTW(gamma=0.3)(*(scale * x for x in c["inputs"]), beta=0.5)
    ds.append(d)
    ps.append(p)
c["distance"], c["penalty"] = torch.stack(ds), torch.stack(ps)


# --------------------------------------------------- JEANIE / soft-DTW / FVM

def value_and_grad(fn, x):
    x = x.clone().requires_grad_(True)
    out = fn(x)
    out.sum().backward()
    return out.detach(), x.grad


REF_COST = {"euclidean": J.euclidean_cost, "sqeuclidean": J.squared_euclidean_cost,
            "rbf": J.rbf_cost}

for T, Uu in G.SDTW_SHAPES:
    for gamma in G.GAMMAS:
        c, g = case("jeanie.soft_dtw", T=T, U=Uu, gamma=gamma)
        c["cost"] = torch.rand(T, Uu, generator=g, **f64)
        c["value"], c["grad"] = value_and_grad(lambda x: J.soft_dtw(x, gamma), c["cost"])

for K, T, Uu in G.JEANIE_1D_SHAPES:
    for shift in G.JEANIE_1D_SHIFTS:
        c, g = case("jeanie.jeanie_1d", K=K, T=T, U=Uu, shift=shift)
        c["cost"] = cost = torch.rand(K, T, Uu, generator=g, **f64)
        c["value"], c["grad"] = value_and_grad(lambda x: J.jeanie_1d_from_cost(x, 0.1, shift), cost)
        c["accumulator"] = J.jeanie_1d_from_cost(cost, 0.1, shift, return_accumulator=True)[1]

for K1, K2, T, Uu in G.JEANIE_2D_SHAPES:
    for s_az, s_alt in G.JEANIE_2D_SHIFTS:
        c, g = case("jeanie.jeanie_2d", K1=K1, K2=K2, T=T, U=Uu, s_az=s_az, s_alt=s_alt)
        c["cost"] = torch.rand(K1, K2, T, Uu, generator=g, **f64)
        c["value"], c["grad"] = value_and_grad(
            lambda x: J.jeanie_2d_from_cost(x, 0.1, s_az, s_alt), c["cost"])

c, g = case("jeanie.batched")
c["cost_1d"] = torch.rand(6, 4, 7, 9, generator=g, **f64)
c["value_1d"] = torch.stack([J.jeanie_1d_from_cost(x, 0.1, 1) for x in c["cost_1d"]])
c["cost_2d"] = torch.rand(3, 3, 3, 5, 6, generator=g, **f64)
c["value_2d"] = torch.stack([J.jeanie_2d_from_cost(x, 0.1, 1, 1) for x in c["cost_2d"]])

for name, shape in G.FVM_CASES:
    c, g = case("jeanie.fvm", name=name)
    c["cost"] = torch.rand(*shape, generator=g, **f64)
    c["value"], c["grad"] = value_and_grad(lambda x: getattr(J, name)(x, 0.1), c["cost"])
    c["batch"] = torch.rand(3, *shape, generator=g, **f64)
    c["batch_value"] = torch.stack([getattr(J, name)(x, 0.1) for x in c["batch"]])

c, g = case("jeanie.softmin")
c["v"] = v = torch.randn(4, 5, generator=g, **f64)
c["outputs"] = [J.softmin(v, 0.3, dim=1), J.softmin(v, 0.3)]

for name in G.DISTANCES:
    c, g = case("jeanie.distance", name=name)
    q = torch.randn(4, 5, 8, generator=g, **f64).requires_grad_()
    s = torch.randn(6, 8, generator=g, **f64).requires_grad_()
    out = getattr(J, name)(q, s)
    c["q"], c["s"], c["value"] = q.detach(), s.detach(), out.detach()
    c["grads"] = list(torch.autograd.grad(out.sum(), (q, s)))
    c["qb"] = torch.randn(3, 4, 5, 8, generator=g, **f64)
    c["sb"] = torch.randn(3, 6, 8, generator=g, **f64)
    c["batch_value"] = torch.stack([getattr(J, name)(a, b) for a, b in zip(c["qb"], c["sb"])])

c, g = case("jeanie.end_to_end")
q = torch.randn(5, 6, 16, generator=g, **f64).requires_grad_()
s = torch.randn(7, 16, generator=g, **f64).requires_grad_()
d = J.jeanie_1d_from_cost(J.euclidean_cost(q, s), 0.1, 1)
c["q"], c["s"], c["value"] = q.detach(), s.detach(), d.detach()
c["grads"] = list(torch.autograd.grad(d, (q, s)))

for metric in G.METRICS:
    for K, T, Uu, shift in G.FEATURES_1D:
        c, g = case("jeanie.features_1d", metric=metric, K=K, T=T, U=Uu, shift=shift)
        q = torch.randn(2, K, T, 6, generator=g, **f64).requires_grad_()
        s = torch.randn(2, Uu, 6, generator=g, **f64).requires_grad_()
        d = torch.stack([J.jeanie_1d_from_cost(REF_COST[metric](a, b), 0.1, shift)
                         for a, b in zip(q, s)])
        c["q"], c["s"], c["value"] = q.detach(), s.detach(), d.detach()
        c["grads"] = list(torch.autograd.grad(d.sum(), (q, s)))

for metric in ("euclidean", "sqeuclidean"):
    c, g = case("jeanie.features_2d", metric=metric)
    q = torch.randn(3, 3, 2, 5, 4, generator=g, **f64).requires_grad_()
    s = torch.randn(3, 7, 4, generator=g, **f64).requires_grad_()
    cost = lambda a, b: REF_COST[metric](a.reshape(6, 5, 4), b).reshape(3, 2, 5, 7)  # noqa: E731
    d = torch.stack([J.jeanie_2d_from_cost(cost(a, b), 0.1, 1, 1) for a, b in zip(q, s)])
    c["q"], c["s"], c["value"] = q.detach(), s.detach(), d.detach()
    c["grads"] = list(torch.autograd.grad(d.sum(), (q, s)))
    c["accumulator"] = J.jeanie_2d_from_cost(cost(q[0], s[0]), 0.1, 1, 1,
                                             return_accumulator=True)[1].detach()


# Losses that use both the distance and the accumulator (official functions
# take one unbatched pair).

def cost_loss(cost, shift):
    d, R = J.jeanie_1d_from_cost(cost, 0.3, shift, return_accumulator=True)
    return d + 0.1 * (R ** 2).mean()


def feat_loss(q, s, metric, shift=1):
    return cost_loss(REF_COST[metric](q, s), shift)


for K, T, Uu, shift in G.ACCUMULATOR_COST:
    c, g = case("parity.accumulator_cost", K=K, T=T, U=Uu, shift=shift)
    c["cost"] = cost = torch.rand(K, T, Uu, generator=g, **f64)
    x = cost.clone().requires_grad_()
    c["grad"] = torch.autograd.grad(cost_loss(x, shift), x)[0]

c, g = case("parity.accumulator_2d")
c["cost"] = cost = torch.rand(3, 2, 5, 6, generator=g, **f64)
x = cost.clone().requires_grad_()
d, R = J.jeanie_2d_from_cost(x, 0.3, 1, 1, return_accumulator=True)
c["grad"] = torch.autograd.grad(d + R.sin().sum(), x)[0]

for metric in ("euclidean", "sqeuclidean"):
    c, g = case("parity.accumulator_features", metric=metric)
    c["q"] = torch.randn(4, 6, 3, generator=g, **f64)
    c["s"] = torch.randn(5, 3, generator=g, **f64)
    q, s = c["q"].clone().requires_grad_(), c["s"].clone().requires_grad_()
    c["grads"] = list(torch.autograd.grad(feat_loss(q, s, metric), (q, s)))

for path in ("cost", "euclidean", "sqeuclidean"):
    c, g = case("parity.maml", path=path)
    c["q"] = q = torch.randn(4, 5, 3, generator=g, **f64)
    c["s"] = s = torch.randn(6, 3, generator=g, **f64)
    c["W0"] = W0 = torch.randn(3, 3, generator=g, **f64)
    loss = ((lambda a, b: cost_loss(J.euclidean_cost(a, b), 1)) if path == "cost"
            else (lambda a, b: feat_loss(a, b, path)))
    w = W0.clone().requires_grad_()
    gw, = torch.autograd.grad(loss(q @ w.T, s @ w.T), w, create_graph=True)
    w2 = w - 0.1 * gw
    c["meta_grad"] = torch.autograd.grad(loss(q @ w2.T, s @ w2.T), w)[0]

for path in ("cost", "euclidean"):
    c, g = case("parity.func", path=path)
    q = torch.randn(3, 4, 2, generator=g, **f64)
    c["s"] = s = torch.randn(4, 2, generator=g, **f64)
    if path == "cost":
        x, f = J.euclidean_cost(q, s), (lambda cc: cost_loss(cc, 1))
    else:
        x, f = q, (lambda qq: feat_loss(qq, s, path))
    c["x"] = x
    c["v"] = v = torch.randn(x.shape, generator=g, **f64)
    c["jacobian"] = torch.autograd.functional.jacobian(f, x)
    c["jvp"] = torch.autograd.functional.jvp(f, x, v)[1]
    c["hessian"] = torch.autograd.functional.hessian(f, x)

c, g = case("parity.vmap")
c["cost"] = cost = torch.rand(3, 2, 4, 5, 6, generator=g, **f64)   # [V, B, K, T, U]
c["cost_value"] = torch.stack([torch.stack([J.jeanie_1d_from_cost(cost[v, b], 0.3, 1)
                                            for b in range(2)]) for v in range(3)])
c["q"] = q = torch.randn(3, 2, 4, 5, 3, generator=g, **f64)
c["s"] = s = torch.randn(3, 2, 6, 3, generator=g, **f64)
c["features_value"] = torch.stack([torch.stack([
    J.jeanie_1d_from_cost(J.euclidean_cost(q[v, b], s[v, b]), 0.3, 1) for b in range(2)])
    for v in range(3)])


# Second-order gradients without the accumulator: the distance alone, as a
# function of one input x (a cost tensor, or the query features).

def second_order_fn(method, s):
    if method == "jeanie_cost":
        return lambda x: J.jeanie_1d_from_cost(x, 0.3, 1)
    if method == "jeanie_features":
        return lambda x: J.jeanie_1d_from_cost(J.euclidean_cost(x, s), 0.3, 1)
    if method == "fvm":
        return lambda x: J.fvm_query_only_1d(x, 0.3)
    return lambda x: J.soft_dtw(x, 0.3)


def sq_dist(a, b):
    """[T, D], [U, D] -> [T, U] squared Euclidean distances (plain torch ops)."""
    return ((a[:, None] - b[None]) ** 2).sum(-1)


def maml_loss(method, a, b):
    """Distance of the transformed features a [K, T, D] (or [T, D]), b [U, D]."""
    if method == "jeanie_cost":
        return J.jeanie_1d_from_cost(J.euclidean_cost(a, b), 0.3, 1)
    if method == "jeanie_features":
        return J.jeanie_1d_from_cost(J.euclidean_cost(a, b), 0.3, 1)
    if method == "fvm":
        return J.fvm_query_only_1d(J.euclidean_cost(a, b), 0.3)
    return J.soft_dtw(sq_dist(a, b), 0.3)


for method in G.SECOND_ORDER:
    c, g = case("parity.second_order", method=method)
    c["s"] = s = torch.randn(5, 2, generator=g, **f64)
    if method == "jeanie_features":
        x = torch.randn(3, 4, 2, generator=g, **f64)
    elif method == "soft_dtw":
        x = torch.rand(4, 5, generator=g, **f64)
    else:
        x = torch.rand(3, 4, 5, generator=g, **f64)
    f = second_order_fn(method, s)
    c["x"] = x
    c["v"] = v = torch.randn(x.shape, generator=g, **f64)
    c["jacobian"] = torch.autograd.functional.jacobian(f, x)
    c["jvp"] = torch.autograd.functional.jvp(f, x, v)[1]
    c["hessian"] = torch.autograd.functional.hessian(f, x)

    c2, g = case("parity.maml_without_accumulator", method=method)
    shape = (4, 3) if method == "soft_dtw" else (4, 4, 3)
    c2["q"] = q = torch.randn(*shape, generator=g, **f64)
    c2["s"] = s = torch.randn(6, 3, generator=g, **f64)
    c2["W0"] = W0 = torch.randn(3, 3, generator=g, **f64)
    w = W0.clone().requires_grad_()
    gw, = torch.autograd.grad(maml_loss(method, q @ w.T, s @ w.T), w, create_graph=True)
    w2 = w - 0.1 * gw
    c2["meta_grad"] = torch.autograd.grad(maml_loss(method, q @ w2.T, s @ w2.T), w)[0]


def _detach_all(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone().contiguous()
    if isinstance(x, (list, tuple)):
        return [_detach_all(y) for y in x]
    return {k: _detach_all(v) for k, v in x.items()}


G.PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(_detach_all(OUT), G.PATH)
print("wrote {} cases to {} ({:.0f} KiB)".format(len(OUT), G.PATH, G.PATH.stat().st_size / 1024))
