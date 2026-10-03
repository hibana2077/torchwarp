"""torchwarp uDTW vs the official implementation (float64)."""

import pytest
import torch

from conftest import reference
from torchwarp import uDTW, pairwise_matrices, udtw_from_matrices
from torchwarp import _cuda

_ref = reference("udtw")
RefUDTW = _ref.uDTW
ref_pairwise = _ref.pairwise_matrices

HAS_CUDA = _cuda.available("udtw")
BACKENDS = [("torch", "cpu")]
if torch.cuda.is_available():
    BACKENDS.append(("torch", "cuda"))
if HAS_CUDA:
    BACKENDS.append(("cuda", "cuda"))
IDS = ["{}-{}".format(b, d) for b, d in BACKENDS]


def _inputs(B, N, M, D=5, dtype=torch.float64):
    return (
        torch.rand(B, N, D, dtype=dtype),
        torch.rand(B, M, D, dtype=dtype),
        0.5 + torch.rand(B, N, 1, dtype=dtype),
        0.5 + torch.rand(B, M, 1, dtype=dtype),
    )


def _run(module, inputs, device, beta=0.7):
    xs = [x.detach().to(device).requires_grad_(True) for x in inputs]
    dist, pen = module(*xs, beta=beta)
    # distinct weights so both outputs' gradients are exercised
    (dist.sum() + 2.0 * pen.sum()).backward()
    return [dist.detach().cpu(), pen.detach().cpu()] + [x.grad.cpu() for x in xs]


def _assert_all_close(got, want):
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("N,M", [(1, 1), (1, 6), (6, 1), (5, 8), (8, 5), (9, 9)])
@pytest.mark.parametrize("gamma", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("normalize", [False, True])
def test_matches_reference(backend, device, N, M, gamma, normalize):
    inputs = _inputs(3, N, M)
    want = _run(RefUDTW(gamma=gamma, normalize=normalize), inputs, "cpu")
    got = _run(uDTW(gamma=gamma, normalize=normalize, backend=backend), inputs, device)
    _assert_all_close(got, want)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("N,M,bandwidth", [(9, 9, 1), (9, 9, 2), (6, 10, 4), (10, 6, 4.5), (7, 7, 0)])
def test_bandwidth_matches_reference(backend, device, N, M, bandwidth):
    inputs = _inputs(2, N, M)
    want = _run(RefUDTW(gamma=0.1, bandwidth=bandwidth), inputs, "cpu")
    got = _run(uDTW(gamma=0.1, bandwidth=bandwidth, backend=backend), inputs, device)
    _assert_all_close(got, want)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_unreachable_band_raises(backend, device):
    inputs = [x.to(device) for x in _inputs(1, 3, 9)]
    with pytest.raises(ValueError):
        uDTW(gamma=0.1, bandwidth=2, backend=backend)(*inputs)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("beta", [0.0, 1.0, 3.0])
def test_beta(backend, device, beta):
    inputs = _inputs(2, 4, 6)
    want = _run(RefUDTW(gamma=0.1), inputs, "cpu", beta=beta)
    got = _run(uDTW(gamma=0.1, backend=backend), inputs, device, beta=beta)
    _assert_all_close(got, want)


def test_pairwise_matrices_match_reference():
    X, Y, SX, SY = _inputs(2, 4, 6)
    for method in ("gemm", "diff"):
        got = pairwise_matrices(X, Y, SX, SY, beta=0.5, method=method)
        want = ref_pairwise(X, Y, SX, SY, beta=0.5)
        for a, b in zip(got, want):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_gradcheck(backend, device):
    kw = dict(dtype=torch.float64, device=device, requires_grad=True)
    cost = torch.rand(2, 4, 5, **kw)
    pen = torch.randn(2, 4, 5, **kw)

    def fn(c, p):
        d, q = udtw_from_matrices(c, p, gamma=0.3, backend=backend)
        return d, q

    assert torch.autograd.gradcheck(fn, (cost, pen))


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_gradcheck_with_band(backend, device):
    kw = dict(dtype=torch.float64, device=device, requires_grad=True)
    cost = torch.rand(2, 6, 6, **kw)
    pen = torch.randn(2, 6, 6, **kw)
    fn = lambda c, p: udtw_from_matrices(c, p, gamma=0.3, bandwidth=1, backend=backend)
    assert torch.autograd.gradcheck(fn, (cost, pen))


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA extension unavailable")
def test_float32_accuracy():
    # float32 vs the float64 truth, as max-normalised error. The reference
    # implementation itself reaches ~2e-4 on the gradients here.
    inputs = _inputs(16, 20, 25, D=32)
    truth = _run(RefUDTW(gamma=0.1), inputs, "cpu")
    inputs32 = [x.float() for x in inputs]
    for backend, device in BACKENDS:
        got = _run(uDTW(gamma=0.1, backend=backend), inputs32, device)
        for a, b in zip(got, truth):
            err = ((a.double() - b).abs().max() / b.abs().max()).item()
            assert err < 1e-3, (backend, device, err)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA extension unavailable")
def test_long_sequences_use_global_scratch():
    # 12 * N * 8 bytes > 48 KiB forces the global-memory buffer path.
    cost = torch.rand(2, 600, 610, dtype=torch.float64)
    pen = torch.randn(2, 600, 610, dtype=torch.float64)
    a = udtw_from_matrices(cost, pen, gamma=0.1, backend="torch")
    b = udtw_from_matrices(cost.cuda(), pen.cuda(), gamma=0.1, backend="cuda")
    for x, y in zip(b, a):
        torch.testing.assert_close(x.cpu(), y, atol=1e-8, rtol=1e-10)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA extension unavailable")
@pytest.mark.parametrize("N,M,bandwidth,normalize", [
    (12, 15, None, False), (15, 12, None, True), (9, 9, 2, False), (33, 33, None, True),
])
def test_fused_kernel_matches_unfused(N, M, bandwidth, normalize):
    # distance="gemm" on CUDA takes the fused kernel; "diff" the unfused one.
    inputs = _inputs(4, N, M)
    fused = _run(uDTW(gamma=0.1, bandwidth=bandwidth, normalize=normalize,
                      backend="cuda"), inputs, "cuda")
    unfused = _run(uDTW(gamma=0.1, bandwidth=bandwidth, normalize=normalize,
                        backend="cuda", distance="diff"), inputs, "cuda")
    _assert_all_close(fused, unfused)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA extension unavailable")
def test_fused_gradcheck():
    kw = dict(dtype=torch.float64, device="cuda", requires_grad=True)
    X, Y = torch.rand(2, 4, 3, **kw), torch.rand(2, 6, 3, **kw)
    SX = (0.5 + torch.rand(2, 4, 1, dtype=torch.float64, device="cuda")).requires_grad_()
    SY = (0.5 + torch.rand(2, 6, 1, dtype=torch.float64, device="cuda")).requires_grad_()
    module = uDTW(gamma=0.3, backend="cuda")
    assert torch.autograd.gradcheck(lambda *a: module(*a, beta=0.8), (X, Y, SX, SY))


def test_only_one_output_used():
    X, Y, SX, SY = [x.requires_grad_(True) for x in _inputs(2, 4, 5)]
    dist, _ = uDTW(gamma=0.1)(X, Y, SX, SY)
    dist.sum().backward()
    assert torch.isfinite(X.grad).all() and torch.isfinite(SX.grad).all()


def test_input_validation():
    X, Y, SX, SY = _inputs(2, 4, 5)
    with pytest.raises(ValueError):
        uDTW(gamma=0.0)
    with pytest.raises(ValueError):
        uDTW(gamma=0.1)(X, Y, SX, SY, beta=-1.0)
    with pytest.raises(ValueError):
        uDTW(gamma=0.1)(X[0], Y, SX, SY)
    with pytest.raises(ValueError):
        uDTW(gamma=0.1, backend="nope")
