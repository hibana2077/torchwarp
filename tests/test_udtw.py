"""torchwarp uDTW vs the stored outputs of the official implementation (float64)."""

import pytest
import torch

from torchwarp import UDTW, _cuda
from torchwarp.functional import pairwise_matrices, softmin, softmin_weights, udtw_from_cost

import golden as G
from golden import golden

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


def _run(fn, inputs, device):
    xs = [x.detach().to(device).requires_grad_(True) for x in inputs]
    dist, pen = fn(*xs)
    # distinct weights so both outputs' gradients are exercised
    (dist.sum() + 2.0 * pen.sum()).backward()
    return [dist.detach().cpu(), pen.detach().cpu()] + [x.grad.cpu() for x in xs]


def _assert_all_close(got, want):
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("N,M", G.UDTW_SHAPES)
@pytest.mark.parametrize("gamma", G.GAMMAS)
@pytest.mark.parametrize("normalize", [False, True])
def test_matches_reference(backend, device, N, M, gamma, normalize):
    ref = golden("udtw.matches", N=N, M=M, gamma=gamma, normalize=normalize)
    got = _run(UDTW(gamma=gamma, beta=0.7, normalize=normalize, backend=backend),
               ref["inputs"], device)
    _assert_all_close(got, ref["outputs"])


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("N,M,bandwidth", G.UDTW_BANDS)
def test_bandwidth_matches_reference(backend, device, N, M, bandwidth):
    ref = golden("udtw.bandwidth", N=N, M=M, bandwidth=bandwidth)
    got = _run(UDTW(gamma=0.1, beta=0.7, bandwidth=bandwidth, backend=backend),
               ref["inputs"], device)
    _assert_all_close(got, ref["outputs"])


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_unreachable_band_raises(backend, device):
    inputs = [x.to(device) for x in _inputs(1, 3, 9)]
    with pytest.raises(ValueError):
        UDTW(gamma=0.1, bandwidth=2, backend=backend)(*inputs)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("beta", G.UDTW_BETAS)
def test_beta(backend, device, beta):
    ref = golden("udtw.beta", beta=beta)
    got = _run(UDTW(gamma=0.1, beta=beta, backend=backend), ref["inputs"], device)
    _assert_all_close(got, ref["outputs"])


def test_pairwise_matrices_match_reference():
    ref = golden("udtw.pairwise_matrices")
    for method in ("gemm", "diff"):
        got = pairwise_matrices(*ref["inputs"], beta=0.5, method=method)
        for a, b in zip(got, ref["outputs"]):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)


def test_softmin_matches_reference():
    ref = golden("udtw.softmin")
    v, want = ref["v"], ref["outputs"]
    torch.testing.assert_close(softmin(v, 0.3, dim=1), want[0])
    torch.testing.assert_close(softmin(v, 0.3), want[1])
    torch.testing.assert_close(softmin_weights(v, 0.3, dim=1), want[2])


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_gradcheck(backend, device):
    kw = dict(dtype=torch.float64, device=device, requires_grad=True)
    cost = torch.rand(2, 4, 5, **kw)
    pen = torch.randn(2, 4, 5, **kw)

    def fn(c, p):
        d, q = udtw_from_cost(c, p, gamma=0.3, backend=backend)
        return d, q

    assert torch.autograd.gradcheck(fn, (cost, pen))


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_gradcheck_with_band(backend, device):
    kw = dict(dtype=torch.float64, device=device, requires_grad=True)
    cost = torch.rand(2, 6, 6, **kw)
    pen = torch.randn(2, 6, 6, **kw)
    fn = lambda c, p: udtw_from_cost(c, p, gamma=0.3, bandwidth=1, backend=backend)
    assert torch.autograd.gradcheck(fn, (cost, pen))


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA extension unavailable")
def test_float32_accuracy():
    # float32 vs the float64 truth, as max-normalised error. The reference
    # implementation itself reaches ~2e-4 on the gradients here.
    ref = golden("udtw.float32")
    truth = ref["outputs"]
    inputs32 = [x.float() for x in ref["inputs"]]
    for backend, device in BACKENDS:
        got = _run(UDTW(gamma=0.1, beta=0.7, backend=backend), inputs32, device)
        for a, b in zip(got, truth):
            err = ((a.double() - b).abs().max() / b.abs().max()).item()
            assert err < 1e-3, (backend, device, err)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA extension unavailable")
def test_long_sequences_use_global_scratch():
    # 12 * N * 8 bytes > 48 KiB forces the global-memory buffer path.
    cost = torch.rand(2, 600, 610, dtype=torch.float64)
    pen = torch.randn(2, 600, 610, dtype=torch.float64)
    a = udtw_from_cost(cost, pen, gamma=0.1, backend="torch")
    b = udtw_from_cost(cost.cuda(), pen.cuda(), gamma=0.1, backend="cuda")
    for x, y in zip(b, a):
        torch.testing.assert_close(x.cpu(), y, atol=1e-8, rtol=1e-10)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA extension unavailable")
@pytest.mark.parametrize("N,M,bandwidth,normalize", [
    (12, 15, None, False), (15, 12, None, True), (9, 9, 2, False), (33, 33, None, True),
])
def test_fused_kernel_matches_unfused(N, M, bandwidth, normalize):
    # method="gemm" on CUDA takes the fused kernel; "diff" the unfused one.
    inputs = _inputs(4, N, M)
    fused = _run(UDTW(gamma=0.1, beta=0.7, bandwidth=bandwidth, normalize=normalize,
                      backend="cuda"), inputs, "cuda")
    unfused = _run(UDTW(gamma=0.1, beta=0.7, bandwidth=bandwidth, normalize=normalize,
                        backend="cuda", method="diff"), inputs, "cuda")
    _assert_all_close(fused, unfused)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA extension unavailable")
def test_fused_gradcheck():
    kw = dict(dtype=torch.float64, device="cuda", requires_grad=True)
    X, Y = torch.rand(2, 4, 3, **kw), torch.rand(2, 6, 3, **kw)
    SX = (0.5 + torch.rand(2, 4, 1, dtype=torch.float64, device="cuda")).requires_grad_()
    SY = (0.5 + torch.rand(2, 6, 1, dtype=torch.float64, device="cuda")).requires_grad_()
    module = UDTW(gamma=0.3, beta=0.8, backend="cuda")
    assert torch.autograd.gradcheck(module, (X, Y, SX, SY))


def test_only_one_output_used():
    X, Y, SX, SY = [x.requires_grad_(True) for x in _inputs(2, 4, 5)]
    dist, _ = UDTW(gamma=0.1)(X, Y, SX, SY)
    dist.sum().backward()
    assert torch.isfinite(X.grad).all() and torch.isfinite(SX.grad).all()


def test_input_validation():
    X, Y, SX, SY = _inputs(2, 4, 5)
    with pytest.raises(ValueError):
        UDTW(gamma=0.0)
    with pytest.raises(ValueError):
        UDTW(gamma=0.1, beta=-1.0)
    with pytest.raises(ValueError):
        UDTW(gamma=0.1)(X[0], Y, SX, SY)
    with pytest.raises(ValueError):           # unbatched inputs are not accepted
        UDTW(gamma=0.1)(X[0], Y[0], SX[0], SY[0])
    with pytest.raises(ValueError):
        UDTW(gamma=0.1, backend="nope")
