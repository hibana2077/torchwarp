"""Feature parity with the reference (pure-autograd) implementation.

Everything the reference supports must work, with the same numbers:
higher-order gradients (create_graph / MAML), forward mode, every
torch.func transform, torch.compile and autocast.
"""

import pytest
import torch

from conftest import reference
from torchwarp import uDTW, pairwise_matrices, udtw_from_features, udtw_from_matrices
from torchwarp import _cuda

RefUDTW = reference("udtw").uDTW

HAS_CUDA = _cuda.available("udtw")
BACKENDS = [("torch", "cpu")]
if torch.cuda.is_available():
    BACKENDS.append(("torch", "cuda"))
if HAS_CUDA:
    BACKENDS.append(("cuda", "cuda"))
IDS = ["{}-{}".format(b, d) for b, d in BACKENDS]
DISTANCES = ["gemm", "diff"]  # fused op / DP op on materialised matrices

f64 = dict(dtype=torch.float64)


def _close(a, b, tol=1e-9):
    torch.testing.assert_close(a, b, atol=tol, rtol=tol)


def _inputs(B=2, N=5, M=6, D=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(B, N, D, generator=g, **f64), torch.rand(B, M, D, generator=g, **f64),
            0.5 + torch.rand(B, N, 1, generator=g, **f64),
            0.5 + torch.rand(B, M, 1, generator=g, **f64))


def _scalar(module, beta=0.8):
    def f(X, Y, SX, SY):
        d, p = module(X, Y, SX, SY, beta=beta)
        return d.sum() + 2.0 * p.sum()
    return f


# ----------------------------------------------------- higher-order gradients

@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("normalize,bandwidth", [(False, None), (True, None), (False, 2)])
def test_maml_meta_gradient(backend, device, distance, normalize, bandwidth):
    X, Y, SX, SY = _inputs()
    W0 = torch.randn(4, 4, generator=torch.Generator().manual_seed(1), **f64)

    def meta_grad(module, dev):
        w = W0.clone().to(dev).requires_grad_()
        a, b, sa, sb = (t.to(dev) for t in (X, Y, SX, SY))
        f = _scalar(module)
        g, = torch.autograd.grad(f(a @ w.T, b @ w.T, sa, sb), w, create_graph=True)
        w2 = w - 0.1 * g
        return torch.autograd.grad(f(a @ w2.T, b @ w2.T, sa, sb), w)[0].cpu()

    kw = dict(gamma=0.3, normalize=normalize, bandwidth=bandwidth)
    want = meta_grad(RefUDTW(**kw), "cpu")
    got = meta_grad(uDTW(backend=backend, distance=distance, **kw), device)
    _close(got, want)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("distance", DISTANCES)
def test_third_derivative(backend, device, distance):
    X, Y, SX, SY = _inputs(seed=3)
    V = torch.randn_like(X)

    def d3(module, dev):
        t = torch.zeros((), device=dev, **f64).requires_grad_()
        f = _scalar(module)
        y = f(X.to(dev) + t * V.to(dev), Y.to(dev), SX.to(dev), SY.to(dev))
        for _ in range(3):
            y, = torch.autograd.grad(y, t, create_graph=True)
        return y.detach().cpu()

    _close(d3(uDTW(gamma=0.5, backend=backend, distance=distance), device),
           d3(RefUDTW(gamma=0.5), "cpu"), tol=1e-7)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_gradgradcheck(backend, device):
    kw = dict(device=device, **f64)
    cost = torch.rand(2, 3, 4, **kw).requires_grad_()
    pen = torch.randn(2, 3, 4, **kw).requires_grad_()
    f = lambda c, p: udtw_from_matrices(c, p, gamma=0.4, backend=backend)
    assert torch.autograd.gradgradcheck(f, (cost, pen))

    X, Y, SX, SY = (t.to(device).requires_grad_() for t in _inputs(B=1, N=3, M=4, D=2))
    g = lambda *a: udtw_from_features(*a, beta=0.7, gamma=0.4, backend=backend)
    assert torch.autograd.gradgradcheck(g, (X, Y, SX, SY))


# ------------------------------------------------------------- torch.func

def _ref_loss(X):
    _, Y, SX, SY = _inputs(B=1, N=3, M=4, D=2)
    return _scalar(RefUDTW(gamma=0.4))(X, Y, SX, SY)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("distance", DISTANCES)
def test_torch_func_transforms(backend, device, distance):
    X, Y, SX, SY = _inputs(B=1, N=3, M=4, D=2)
    rest = tuple(t.to(device) for t in (Y, SX, SY))
    fast = uDTW(gamma=0.4, backend=backend, distance=distance)
    f = lambda x: _scalar(fast)(x, *rest)
    x = X.to(device)

    _close(torch.func.grad(f)(x).cpu(), torch.autograd.functional.jacobian(_ref_loss, X))
    _close(torch.func.jacrev(f)(x).cpu(), torch.autograd.functional.jacobian(_ref_loss, X))
    _close(torch.func.jacfwd(f)(x).cpu(), torch.autograd.functional.jacobian(_ref_loss, X))
    v = torch.randn_like(X)
    _, jvp = torch.func.jvp(f, (x,), (v.to(device),))
    _, ref_jvp = torch.autograd.functional.jvp(_ref_loss, X, v)
    _close(jvp.cpu(), ref_jvp)
    H = torch.autograd.functional.hessian(_ref_loss, X)
    _close(torch.func.hessian(f)(x).cpu(), H, tol=1e-7)
    _close(torch.func.jacrev(torch.func.jacrev(f))(x).cpu(), H, tol=1e-7)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("distance", DISTANCES)
def test_vmap(backend, device, distance):
    X, Y, SX, SY = (t.to(device) for t in _inputs(B=2, N=4, M=5))
    stack = lambda t: torch.stack([t, 1.1 * t, 0.9 * t])
    fast = uDTW(gamma=0.3, backend=backend, distance=distance)
    d, p = torch.func.vmap(lambda *a: fast(*a, beta=0.5))(stack(X), stack(Y), stack(SX), stack(SY))
    for k in range(3):
        dk, pk = RefUDTW(gamma=0.3)(*(stack(t)[k].cpu() for t in (X, Y, SX, SY)), beta=0.5)
        _close(d[k].cpu(), dk)
        _close(p[k].cpu(), pk)
    # per-sample gradients: vmap(grad)
    g = torch.func.vmap(torch.func.grad(lambda x, y, sx, sy: _scalar(fast)(x[None], y[None], sx[None], sy[None])))
    per_sample = g(X, Y, SX, SY)
    full = torch.func.grad(_scalar(fast))(X, Y, SX, SY)
    _close(per_sample, full)


# ------------------------------------------------- torch.compile and autocast

@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("distance", DISTANCES)
def test_torch_compile_fullgraph(backend, device, distance):
    torch._dynamo.reset()
    X, Y, SX, SY = (t.to(device).requires_grad_() for t in _inputs(B=3, N=5, M=7))
    fast = uDTW(gamma=0.3, normalize=True, backend=backend, distance=distance)
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
    for distance in DISTANCES:
        m = uDTW(gamma=0.3, distance=distance)
        with torch.autocast("cuda", dtype=torch.float16):
            d, p = m(lin(X), lin(Y), SX, SY)
        assert d.dtype == torch.float32 and p.dtype == torch.float32
        (d.sum() + p.sum()).backward()
        assert torch.isfinite(X.grad).all()
        ref_d, _ = m(lin(X).float(), lin(Y).float(), SX, SY)
        torch.testing.assert_close(d, ref_d, atol=5e-2, rtol=5e-3)
        X.grad = None


def test_pairwise_matrices_path_is_differentiable_twice():
    # distance="diff" builds matrices in PyTorch; gradients of every order
    # flow through pairwise_matrices and the DP op.
    X, Y, SX, SY = (t.requires_grad_() for t in _inputs(B=1, N=3, M=3, D=2))
    f = lambda *a: udtw_from_matrices(*pairwise_matrices(*a, beta=0.5)[:2], gamma=0.4)
    assert torch.autograd.gradgradcheck(f, (X, Y, SX, SY))
