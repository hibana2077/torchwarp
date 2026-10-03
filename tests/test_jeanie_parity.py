"""Feature parity with the official (pure-autograd) implementation.

Everything the official code supports must work, with the same numbers: a
differentiable accumulator, higher-order gradients (create_graph / MAML),
forward mode, every torch.func transform, torch.compile and autocast.
"""

import pytest
import torch

import torchwarp as fast
from conftest import reference
from torchwarp import _cuda

ref = reference("jeanie")

HAS_CUDA = _cuda.available("jeanie")
BACKENDS = [("torch", "cpu")]
if torch.cuda.is_available():
    BACKENDS.append(("torch", "cuda"))
if HAS_CUDA:
    BACKENDS.append(("cuda", "cuda"))
IDS = ["{}-{}".format(b, d) for b, d in BACKENDS]

f64 = dict(dtype=torch.float64)
_REF_COST = {"euclidean": ref.euclidean_cost, "sqeuclidean": ref.squared_euclidean_cost}


def _close(a, b, tol=1e-9):
    torch.testing.assert_close(a, b, atol=tol, rtol=tol)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


# Losses that use both the distance and the accumulator, for one pair
# (reference) and for a batch (fast). "cost" variants take a cost tensor,
# "feat" variants take query/support features.

def _ref_cost_loss(cost, shift):
    d, R = ref.jeanie_1d_from_cost(cost, 0.3, shift, return_accumulator=True)
    return d + 0.1 * (R ** 2).mean()


def _fast_cost_loss(cost, shift, backend):
    d, R = fast.jeanie_1d_from_cost(cost, 0.3, shift, return_accumulator=True,
                                    backend=backend)
    return d + 0.1 * (R ** 2).mean()


def _ref_feat_loss(q, s, metric, shift=1):
    d, R = ref.jeanie_1d_from_cost(_REF_COST[metric](q, s), 0.3, shift,
                                   return_accumulator=True)
    return d + 0.1 * (R ** 2).mean()


def _fast_feat_loss(q, s, metric, backend, shift=1):
    d, R = fast.jeanie_1d_from_features(q, s, 0.3, shift, metric=metric,
                                        return_accumulator=True, backend=backend)
    return d + 0.1 * (R ** 2).mean()


# --------------------------------------------------- differentiable accumulator

@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("K,T,U,shift", [(4, 5, 7, 1), (3, 7, 4, 2), (1, 4, 4, 0)])
def test_accumulator_gradient_cost(backend, device, K, T, U, shift):
    cost = torch.rand(K, T, U, generator=_gen(0), **f64)
    want = torch.autograd.grad(_ref_cost_loss(cost.requires_grad_(), shift), cost)[0]
    c = cost.detach().to(device).requires_grad_()
    got = torch.autograd.grad(_fast_cost_loss(c, shift, backend), c)[0]
    _close(got.cpu(), want)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_accumulator_gradient_2d(backend, device):
    cost = torch.rand(3, 2, 5, 6, generator=_gen(1), **f64)

    def loss(fn, c, **kw):
        d, R = fn(c, 0.3, 1, 1, return_accumulator=True, **kw)
        return d + (R.sin()).sum()

    want = torch.autograd.grad(loss(ref.jeanie_2d_from_cost, cost.requires_grad_()), cost)[0]
    c = cost.detach().to(device).requires_grad_()
    got = torch.autograd.grad(loss(fast.jeanie_2d_from_cost, c, backend=backend), c)[0]
    _close(got.cpu(), want)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean"])
def test_accumulator_gradient_features(backend, device, metric):
    q = torch.randn(4, 6, 3, generator=_gen(2), **f64)
    s = torch.randn(5, 3, generator=_gen(3), **f64)
    qa, sa = q.clone().requires_grad_(), s.clone().requires_grad_()
    want = torch.autograd.grad(_ref_feat_loss(qa, sa, metric), (qa, sa))
    qb, sb = q.to(device).requires_grad_(), s.to(device).requires_grad_()
    got = torch.autograd.grad(_fast_feat_loss(qb, sb, metric, backend), (qb, sb))
    for a, b in zip(got, want):
        _close(a.cpu(), b)


# ----------------------------------------------------- higher-order gradients

@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("path", ["cost", "euclidean", "sqeuclidean"])
def test_maml_meta_gradient(backend, device, path):
    q = torch.randn(4, 5, 3, generator=_gen(4), **f64)
    s = torch.randn(6, 3, generator=_gen(5), **f64)
    W0 = torch.randn(3, 3, generator=_gen(6), **f64)

    def loss(qw, sw, fast_mode, dev):
        if path == "cost":
            cost = ref.euclidean_cost(qw, sw)
            return (_fast_cost_loss(cost, 1, backend) if fast_mode
                    else _ref_cost_loss(cost, 1))
        return (_fast_feat_loss(qw, sw, path, backend) if fast_mode
                else _ref_feat_loss(qw, sw, path))

    def meta_grad(fast_mode, dev):
        w = W0.clone().to(dev).requires_grad_()
        a, b = q.to(dev), s.to(dev)
        g, = torch.autograd.grad(loss(a @ w.T, b @ w.T, fast_mode, dev), w, create_graph=True)
        w2 = w - 0.1 * g
        return torch.autograd.grad(loss(a @ w2.T, b @ w2.T, fast_mode, dev), w)[0].cpu()

    _close(meta_grad(True, device), meta_grad(False, "cpu"))


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_gradgradcheck(backend, device):
    cost = torch.rand(2, 2, 2, 3, 4, device=device, **f64).requires_grad_()
    f = lambda c: fast.jeanie_dp(c, 0.5, 1, 1, backend, return_accumulator=True)
    assert torch.autograd.gradgradcheck(lambda c: tuple(f(c)), (cost,))

    q = torch.randn(1, 2, 1, 3, 2, device=device, **f64).requires_grad_()
    s = torch.randn(1, 4, 2, device=device, **f64).requires_grad_()
    for metric in ("euclidean", "sqeuclidean"):
        g = lambda q, s: tuple(fast.jeanie_dp_features(q, s, 0.5, 1, 0, metric, backend,
                                                       return_accumulator=True))
        assert torch.autograd.gradgradcheck(g, (q, s))


# ------------------------------------------------------------- torch.func

@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("path", ["cost", "euclidean"])
def test_torch_func_transforms(backend, device, path):
    q = torch.randn(3, 4, 2, generator=_gen(7), **f64)
    s = torch.randn(4, 2, generator=_gen(8), **f64)
    if path == "cost":
        x = ref.euclidean_cost(q, s)
        ref_f = lambda c: _ref_cost_loss(c, 1)
        fast_f = lambda c: _fast_cost_loss(c, 1, backend)
    else:
        x = q
        ref_f = lambda qq: _ref_feat_loss(qq, s, path)
        s_dev = s.to(device)
        fast_f = lambda qq: _fast_feat_loss(qq, s_dev, path, backend)
    xd = x.to(device)

    J = torch.autograd.functional.jacobian(ref_f, x)
    _close(torch.func.grad(fast_f)(xd).cpu(), J)
    _close(torch.func.jacrev(fast_f)(xd).cpu(), J)
    _close(torch.func.jacfwd(fast_f)(xd).cpu(), J)
    v = torch.randn_like(x)
    _, jvp = torch.func.jvp(fast_f, (xd,), (v.to(device),))
    _close(jvp.cpu(), torch.autograd.functional.jvp(ref_f, x, v)[1])
    H = torch.autograd.functional.hessian(ref_f, x)
    _close(torch.func.hessian(fast_f)(xd).cpu(), H, tol=1e-7)
    _close(torch.func.jacrev(torch.func.jacrev(fast_f))(xd).cpu(), H, tol=1e-7)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_vmap(backend, device):
    cost = torch.rand(3, 2, 4, 5, 6, generator=_gen(9), **f64)   # [V, B, K, T, U]
    got = torch.func.vmap(
        lambda c: fast.jeanie_1d_from_cost(c, 0.3, 1, backend=backend))(cost.to(device))
    for v in range(3):
        for b in range(2):
            _close(got[v, b].cpu(), ref.jeanie_1d_from_cost(cost[v, b], 0.3, 1))

    q = torch.randn(3, 2, 4, 5, 3, generator=_gen(10), **f64)
    s = torch.randn(3, 2, 6, 3, generator=_gen(11), **f64)
    got = torch.func.vmap(lambda a, b: fast.jeanie_1d_from_features(
        a, b, 0.3, 1, backend=backend))(q.to(device), s.to(device))
    for v in range(3):
        for b in range(2):
            want = ref.jeanie_1d_from_cost(ref.euclidean_cost(q[v, b], s[v, b]), 0.3, 1)
            _close(got[v, b].cpu(), want)

    # per-sample gradients: vmap(grad)
    f = lambda a, b: fast.jeanie_1d_from_features(a, b, 0.3, 1, backend=backend)
    per = torch.func.vmap(torch.func.grad(f))(q[0].to(device), s[0].to(device))
    full = torch.func.grad(lambda a, b: f(a, b).sum())(q[0].to(device), s[0].to(device))
    _close(per, full)


# ------------------------------------------------- torch.compile and autocast

@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("path", ["cost", "euclidean"])
def test_torch_compile_fullgraph(backend, device, path):
    torch._dynamo.reset()
    q = torch.randn(2, 3, 5, 4, device=device, **f64).requires_grad_()
    s = torch.randn(2, 6, 4, device=device, **f64).requires_grad_()
    if path == "cost":
        def f(q, s):
            d, R = fast.jeanie_1d_from_cost(fast.euclidean_cost(q, s), 0.3, 1,
                                            return_accumulator=True, backend=backend)
            return d.sum() + 0.1 * (R ** 2).mean()
    else:
        def f(q, s):
            d, R = fast.jeanie_1d_from_features(q, s, 0.3, 1, return_accumulator=True,
                                                backend=backend)
            return d.sum() + 0.1 * (R ** 2).mean()
    compiled = torch.compile(f, fullgraph=True)
    want, got = f(q, s), compiled(q, s)
    _close(got, want)
    for a, b in zip(torch.autograd.grad(got, (q, s)), torch.autograd.grad(want, (q, s))):
        _close(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_autocast_runs_in_float32():
    q = torch.randn(2, 3, 5, 8, device="cuda").requires_grad_()
    s = torch.randn(2, 6, 8, device="cuda")
    lin = torch.nn.Linear(8, 8).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        a = fast.jeanie_1d_from_features(lin(q), lin(s), 0.3, 1)
        b = fast.jeanie_1d_from_cost(fast.euclidean_cost(lin(q), lin(s)), 0.3, 1)
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    (a.sum() + b.sum()).backward()
    assert torch.isfinite(q.grad).all()
