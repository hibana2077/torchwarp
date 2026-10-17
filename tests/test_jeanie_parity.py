"""Feature parity with the official (pure-autograd) implementation.

Everything the official code supports must work, with the same numbers as
its stored outputs (tests/golden): a
differentiable accumulator, higher-order gradients (create_graph / MAML),
forward mode, every torch.func transform, torch.compile and autocast.
"""

import pytest
import torch

import torchwarp as fast
import torchwarp.functional as TF
from torchwarp import _cuda
from torchwarp.jeanie.functional import _jeanie_dp, _jeanie_dp_features

import golden as G
from golden import golden

HAS_CUDA = _cuda.available("jeanie")
BACKENDS = [("torch", "cpu")]
if torch.cuda.is_available():
    BACKENDS.append(("torch", "cuda"))
if HAS_CUDA:
    BACKENDS.append(("cuda", "cuda"))
IDS = ["{}-{}".format(b, d) for b, d in BACKENDS]

f64 = dict(dtype=torch.float64)


def _close(a, b, tol=1e-9):
    torch.testing.assert_close(a, b, atol=tol, rtol=tol)


# Losses that use both the distance and the accumulator, for one pair (a
# batch of one; the stored official values are for the same losses).
# "cost" variants take a cost tensor, "feat" variants take query/support
# features.

def _fast_cost_loss(cost, shift, backend):
    d, R = TF.jeanie_from_cost(cost[None], gamma=0.3, max_shift=shift, return_accumulator=True,
                               backend=backend)
    return d[0] + 0.1 * (R ** 2).mean()


def _fast_feat_loss(q, s, metric, backend, shift=1):
    d, R = TF.jeanie(q[None], s[None], gamma=0.3, max_shift=shift, metric=metric,
                     return_accumulator=True, backend=backend)
    return d[0] + 0.1 * (R ** 2).mean()


# --------------------------------------------------- differentiable accumulator

@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("K,T,U,shift", G.ACCUMULATOR_COST)
def test_accumulator_gradient_cost(backend, device, K, T, U, shift):
    ref = golden("parity.accumulator_cost", K=K, T=T, U=U, shift=shift)
    c = ref["cost"].to(device).requires_grad_()
    got = torch.autograd.grad(_fast_cost_loss(c, shift, backend), c)[0]
    _close(got.cpu(), ref["grad"])


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_accumulator_gradient_2d(backend, device):
    ref = golden("parity.accumulator_2d")

    def fast_loss(c):
        d, R = TF.jeanie_from_cost(c[None], gamma=0.3, max_shift=(1, 1),
                                   return_accumulator=True, backend=backend)
        return d[0] + (R.sin()).sum()

    c = ref["cost"].to(device).requires_grad_()
    got = torch.autograd.grad(fast_loss(c), c)[0]
    _close(got.cpu(), ref["grad"])


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean"])
def test_accumulator_gradient_features(backend, device, metric):
    ref = golden("parity.accumulator_features", metric=metric)
    qb, sb = ref["q"].to(device).requires_grad_(), ref["s"].to(device).requires_grad_()
    got = torch.autograd.grad(_fast_feat_loss(qb, sb, metric, backend), (qb, sb))
    for a, b in zip(got, ref["grads"]):
        _close(a.cpu(), b)


# ----------------------------------------------------- higher-order gradients

@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("path", ["cost", "euclidean", "sqeuclidean"])
def test_maml_meta_gradient(backend, device, path):
    ref = golden("parity.maml", path=path)

    def loss(qw, sw):
        if path == "cost":
            return _fast_cost_loss(fast.euclidean_cost(qw[None], sw[None])[0], 1, backend)
        return _fast_feat_loss(qw, sw, path, backend)

    w = ref["W0"].to(device).requires_grad_()
    a, b = ref["q"].to(device), ref["s"].to(device)
    g, = torch.autograd.grad(loss(a @ w.T, b @ w.T), w, create_graph=True)
    w2 = w - 0.1 * g
    got = torch.autograd.grad(loss(a @ w2.T, b @ w2.T), w)[0].cpu()
    _close(got, ref["meta_grad"])


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_gradgradcheck(backend, device):
    cost = torch.rand(2, 2, 2, 3, 4, device=device, **f64).requires_grad_()
    f = lambda c: _jeanie_dp(c, 0.5, 1, 1, backend, return_accumulator=True)
    assert torch.autograd.gradgradcheck(lambda c: tuple(f(c)), (cost,))

    q = torch.randn(1, 2, 1, 3, 2, device=device, **f64).requires_grad_()
    s = torch.randn(1, 4, 2, device=device, **f64).requires_grad_()
    for metric in ("euclidean", "sqeuclidean"):
        g = lambda q, s: tuple(_jeanie_dp_features(q, s, 0.5, 1, 0, metric, backend,
                                                       return_accumulator=True))
        assert torch.autograd.gradgradcheck(g, (q, s))


# ------------------------- second order of the distance alone (no accumulator)

def _second_order_fn(method, s, backend):
    """The distance as a function of one input x, as in tests/golden/generate.py."""
    if method == "jeanie_cost":
        return lambda x: TF.jeanie_from_cost(x[None], gamma=0.3, max_shift=1, backend=backend)[0]
    if method == "jeanie_features":
        return lambda x: TF.jeanie(x[None], s[None], gamma=0.3, max_shift=1, backend=backend)[0]
    if method == "fvm":
        return lambda x: TF.fvm_from_cost(x[None], gamma=0.3, backend=backend)[0]
    return lambda x: TF.soft_dtw_from_cost(x[None], gamma=0.3, backend=backend)[0]


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("method", G.SECOND_ORDER)
def test_second_order_without_accumulator(backend, device, method):
    ref = golden("parity.second_order", method=method)
    f = _second_order_fn(method, ref["s"].to(device), backend)
    x = ref["x"].to(device)

    J = ref["jacobian"]
    _close(torch.func.grad(f)(x).cpu(), J)
    _close(torch.func.jacrev(f)(x).cpu(), J)
    _close(torch.func.jacfwd(f)(x).cpu(), J)
    _, jvp = torch.func.jvp(f, (x,), (ref["v"].to(device),))
    _close(jvp.cpu(), ref["jvp"])
    H = ref["hessian"]
    _close(torch.func.hessian(f)(x).cpu(), H, tol=1e-7)
    _close(torch.func.jacrev(torch.func.jacrev(f))(x).cpu(), H, tol=1e-7)
    _close(torch.autograd.functional.hessian(f, x).cpu(), H, tol=1e-7)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("method", G.SECOND_ORDER)
def test_maml_without_accumulator(backend, device, method):
    ref = golden("parity.maml_without_accumulator", method=method)

    def loss(a, b):
        if method == "jeanie_cost":
            cost = fast.euclidean_cost(a[None], b[None])
            return TF.jeanie_from_cost(cost, gamma=0.3, max_shift=1, backend=backend)[0]
        if method == "jeanie_features":
            return TF.jeanie(a[None], b[None], gamma=0.3, max_shift=1, backend=backend)[0]
        if method == "fvm":
            return TF.fvm(a[None], b[None], gamma=0.3, backend=backend)[0]
        cost = ((a[:, None] - b[None]) ** 2).sum(-1)
        return TF.soft_dtw_from_cost(cost[None], gamma=0.3, backend=backend)[0]

    w = ref["W0"].to(device).requires_grad_()
    a, b = ref["q"].to(device), ref["s"].to(device)
    g, = torch.autograd.grad(loss(a @ w.T, b @ w.T), w, create_graph=True)
    w2 = w - 0.1 * g
    got = torch.autograd.grad(loss(a @ w2.T, b @ w2.T), w)[0].cpu()
    _close(got, ref["meta_grad"])


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_gradgradcheck_without_accumulator(backend, device):
    # finite differences, independent of the official code
    kw = dict(device=device, **f64)
    cost = torch.rand(2, 3, 3, 4, **kw).requires_grad_()
    assert torch.autograd.gradgradcheck(
        lambda c: TF.jeanie_from_cost(c, gamma=0.5, max_shift=1, backend=backend), (cost,))
    assert torch.autograd.gradgradcheck(
        lambda c: TF.fvm_from_cost(c, gamma=0.5, backend=backend), (cost,))
    assert torch.autograd.gradgradcheck(
        lambda c: TF.soft_dtw_from_cost(c[:, 0], gamma=0.5, backend=backend), (cost,))

    q = torch.randn(1, 2, 3, 2, **kw).requires_grad_()
    s = torch.randn(1, 4, 2, **kw).requires_grad_()
    for metric in ("euclidean", "sqeuclidean", "rbf"):
        assert torch.autograd.gradgradcheck(
            lambda q, s: TF.jeanie(q, s, gamma=0.5, max_shift=1, metric=metric, backend=backend),
            (q, s))
        assert torch.autograd.gradgradcheck(
            lambda q, s: TF.fvm(q, s, gamma=0.5, metric=metric, backend=backend), (q, s))


# ------------------------------------------------------------- torch.func

@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("path", ["cost", "euclidean"])
def test_torch_func_transforms(backend, device, path):
    ref = golden("parity.func", path=path)
    if path == "cost":
        fast_f = lambda c: _fast_cost_loss(c, 1, backend)
    else:
        s_dev = ref["s"].to(device)
        fast_f = lambda qq: _fast_feat_loss(qq, s_dev, path, backend)
    xd = ref["x"].to(device)

    J = ref["jacobian"]
    _close(torch.func.grad(fast_f)(xd).cpu(), J)
    _close(torch.func.jacrev(fast_f)(xd).cpu(), J)
    _close(torch.func.jacfwd(fast_f)(xd).cpu(), J)
    _, jvp = torch.func.jvp(fast_f, (xd,), (ref["v"].to(device),))
    _close(jvp.cpu(), ref["jvp"])
    H = ref["hessian"]
    _close(torch.func.hessian(fast_f)(xd).cpu(), H, tol=1e-7)
    _close(torch.func.jacrev(torch.func.jacrev(fast_f))(xd).cpu(), H, tol=1e-7)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_vmap(backend, device):
    ref = golden("parity.vmap")
    cost = ref["cost"]                                           # [V, B, K, T, U]
    got = torch.func.vmap(
        lambda c: TF.jeanie_from_cost(c, gamma=0.3, max_shift=1, backend=backend))(cost.to(device))
    _close(got.cpu(), ref["cost_value"])

    q, s = ref["q"], ref["s"]
    got = torch.func.vmap(lambda a, b: TF.jeanie(
        a, b, gamma=0.3, max_shift=1, backend=backend))(q.to(device), s.to(device))
    _close(got.cpu(), ref["features_value"])

    # per-sample gradients: vmap(grad)
    f = lambda a, b: TF.jeanie(a, b, gamma=0.3, max_shift=1, backend=backend)
    per = torch.func.vmap(torch.func.grad(lambda a, b: f(a[None], b[None])[0]))(
        q[0].to(device), s[0].to(device))
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
            d, R = TF.jeanie_from_cost(fast.euclidean_cost(q, s), gamma=0.3, max_shift=1,
                                       return_accumulator=True, backend=backend)
            return d.sum() + 0.1 * (R ** 2).mean()
    else:
        def f(q, s):
            d, R = TF.jeanie(q, s, gamma=0.3, max_shift=1, return_accumulator=True,
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
        a = TF.jeanie(lin(q), lin(s), gamma=0.3, max_shift=1)
        b = TF.jeanie_from_cost(fast.euclidean_cost(lin(q), lin(s)), gamma=0.3, max_shift=1)
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    (a.sum() + b.sum()).backward()
    assert torch.isfinite(q.grad).all()
