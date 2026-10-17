"""Feature parity with the official (pure-autograd) implementation.

Everything the official code supports must work, with the same numbers as
its stored outputs (tests/golden):
higher-order gradients (create_graph / MAML), forward mode, every
torch.func transform, torch.compile and autocast.
"""

import pytest
import torch

import torchwarp.functional as TF
from torchwarp import UDTW, _cuda
from torchwarp.functional import pairwise_matrices, udtw_from_cost

from golden import golden

HAS_CUDA = _cuda.available("udtw")
BACKENDS = [("torch", "cpu")]
if torch.cuda.is_available():
    BACKENDS.append(("torch", "cuda"))
if HAS_CUDA:
    BACKENDS.append(("cuda", "cuda"))
IDS = ["{}-{}".format(b, d) for b, d in BACKENDS]
METHODS = ["gemm", "diff"]  # fused op / DP op on materialised matrices

f64 = dict(dtype=torch.float64)


def _close(a, b, tol=1e-9):
    torch.testing.assert_close(a, b, atol=tol, rtol=tol)


def _inputs(B=2, N=5, M=6, D=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(B, N, D, generator=g, **f64), torch.rand(B, M, D, generator=g, **f64),
            0.5 + torch.rand(B, N, 1, generator=g, **f64),
            0.5 + torch.rand(B, M, 1, generator=g, **f64))


def _scalar(module):
    def f(X, Y, SX, SY):
        d, p = module(X, Y, SX, SY)
        return d.sum() + 2.0 * p.sum()
    return f


# ----------------------------------------------------- higher-order gradients

@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("normalize,bandwidth", [(False, None), (True, None), (False, 2)])
def test_maml_meta_gradient(backend, device, method, normalize, bandwidth):
    ref = golden("udtw.maml", normalize=normalize, bandwidth=bandwidth)
    X, Y, SX, SY = ref["inputs"]
    W0 = ref["W0"]

    def meta_grad(module, dev):
        w = W0.clone().to(dev).requires_grad_()
        a, b, sa, sb = (t.to(dev) for t in (X, Y, SX, SY))
        f = _scalar(module)
        g, = torch.autograd.grad(f(a @ w.T, b @ w.T, sa, sb), w, create_graph=True)
        w2 = w - 0.1 * g
        return torch.autograd.grad(f(a @ w2.T, b @ w2.T, sa, sb), w)[0].cpu()

    got = meta_grad(UDTW(gamma=0.3, beta=0.8, normalize=normalize, bandwidth=bandwidth,
                         backend=backend, method=method), device)
    _close(got, ref["meta_grad"])


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("method", METHODS)
def test_third_derivative(backend, device, method):
    ref = golden("udtw.third_derivative")
    X, Y, SX, SY = ref["inputs"]
    V = ref["V"]

    def d3(module, dev):
        t = torch.zeros((), device=dev, **f64).requires_grad_()
        f = _scalar(module)
        y = f(X.to(dev) + t * V.to(dev), Y.to(dev), SX.to(dev), SY.to(dev))
        for _ in range(3):
            y, = torch.autograd.grad(y, t, create_graph=True)
        return y.detach().cpu()

    _close(d3(UDTW(gamma=0.5, beta=0.8, backend=backend, method=method), device),
           ref["d3"], tol=1e-7)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_gradgradcheck(backend, device):
    kw = dict(device=device, **f64)
    cost = torch.rand(2, 3, 4, **kw).requires_grad_()
    pen = torch.randn(2, 3, 4, **kw).requires_grad_()
    f = lambda c, p: udtw_from_cost(c, p, gamma=0.4, backend=backend)
    assert torch.autograd.gradgradcheck(f, (cost, pen))

    X, Y, SX, SY = (t.to(device).requires_grad_() for t in _inputs(B=1, N=3, M=4, D=2))
    g = lambda *a: TF.udtw(*a, beta=0.7, gamma=0.4, backend=backend)
    assert torch.autograd.gradgradcheck(g, (X, Y, SX, SY))


# ------------------------------------------------------------- torch.func

@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("method", METHODS)
def test_torch_func_transforms(backend, device, method):
    ref = golden("udtw.func")
    X, Y, SX, SY = ref["inputs"]
    rest = tuple(t.to(device) for t in (Y, SX, SY))
    fast = UDTW(gamma=0.4, beta=0.8, backend=backend, method=method)
    f = lambda x: _scalar(fast)(x, *rest)
    x = X.to(device)

    J = ref["jacobian"]
    _close(torch.func.grad(f)(x).cpu(), J)
    _close(torch.func.jacrev(f)(x).cpu(), J)
    _close(torch.func.jacfwd(f)(x).cpu(), J)
    _, jvp = torch.func.jvp(f, (x,), (ref["v"].to(device),))
    _close(jvp.cpu(), ref["jvp"])
    H = ref["hessian"]
    _close(torch.func.hessian(f)(x).cpu(), H, tol=1e-7)
    _close(torch.func.jacrev(torch.func.jacrev(f))(x).cpu(), H, tol=1e-7)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("method", METHODS)
def test_vmap(backend, device, method):
    ref = golden("udtw.vmap")
    X, Y, SX, SY = (t.to(device) for t in ref["inputs"])
    stack = lambda t: torch.stack([t, 1.1 * t, 0.9 * t])
    fast = UDTW(gamma=0.3, beta=0.5, backend=backend, method=method)
    d, p = torch.func.vmap(fast)(stack(X), stack(Y), stack(SX), stack(SY))
    _close(d.cpu(), ref["distance"])
    _close(p.cpu(), ref["penalty"])
    # per-sample gradients: vmap(grad)
    g = torch.func.vmap(torch.func.grad(lambda x, y, sx, sy: _scalar(fast)(x[None], y[None], sx[None], sy[None])))
    per_sample = g(X, Y, SX, SY)
    full = torch.func.grad(_scalar(fast))(X, Y, SX, SY)
    _close(per_sample, full)


# ------------------------------------------------- torch.compile and autocast

@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("method", METHODS)
def test_torch_compile_fullgraph(backend, device, method):
    torch._dynamo.reset()
    X, Y, SX, SY = (t.to(device).requires_grad_() for t in _inputs(B=3, N=5, M=7))
    fast = UDTW(gamma=0.3, beta=0.8, normalize=True, backend=backend, method=method)
    f = _scalar(fast)
    compiled = torch.compile(f, fullgraph=True)
    want = f(X, Y, SX, SY)
    got = compiled(X, Y, SX, SY)
    _close(got, want)
    _close(torch.autograd.grad(got, (X, Y, SX, SY))[0],
           torch.autograd.grad(want, (X, Y, SX, SY))[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_autocast_runs_in_float32():
    X, Y, SX, SY = (t.float().cuda().requires_grad_() for t in _inputs(B=2, N=5, M=6))
    lin = torch.nn.Linear(4, 4).cuda()
    for method in METHODS:
        m = UDTW(gamma=0.3, method=method)
        with torch.autocast("cuda", dtype=torch.float16):
            d, p = m(lin(X), lin(Y), SX, SY)
        assert d.dtype == torch.float32 and p.dtype == torch.float32
        (d.sum() + p.sum()).backward()
        assert torch.isfinite(X.grad).all()
        ref_d, _ = m(lin(X).float(), lin(Y).float(), SX, SY)
        torch.testing.assert_close(d, ref_d, atol=5e-2, rtol=5e-3)
        X.grad = None


def test_pairwise_matrices_path_is_differentiable_twice():
    # method="diff" builds matrices in PyTorch; gradients of every order
    # flow through pairwise_matrices and the DP op.
    X, Y, SX, SY = (t.requires_grad_() for t in _inputs(B=1, N=3, M=3, D=2))
    f = lambda *a: udtw_from_cost(*pairwise_matrices(*a, beta=0.5)[:2], gamma=0.4)
    assert torch.autograd.gradgradcheck(f, (X, Y, SX, SY))
