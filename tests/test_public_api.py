"""Public API checks that do not need the official implementation."""

import pytest
import torch

import torchwarp
import torchwarp.functional as TF
from torchwarp import FVM, JEANIE, UDTW, _cuda
from torchwarp.jeanie.functional import _jeanie_dp

BACKENDS = [("torch", "cpu")]
if torch.cuda.is_available():
    BACKENDS.append(("torch", "cuda"))
if _cuda.available("udtw") and _cuda.available("jeanie"):
    BACKENDS.append(("cuda", "cuda"))
IDS = ["{}-{}".format(b, d) for b, d in BACKENDS]
f64 = dict(dtype=torch.float64)


def test_exports():
    for name in torchwarp.__all__:
        assert hasattr(torchwarp, name), name
    for name in TF.__all__:
        assert hasattr(TF, name), name
    assert torchwarp.__version__
    assert torchwarp.functional is TF
    # removed in the API refactor: the alias and the old top-level functions
    for name in ("uDTW", "udtw_from_features", "jeanie_1d_from_cost", "fvm_query_only_1d",
                 "soft_dtw", "softmin"):
        assert not hasattr(torchwarp, name), name
    with pytest.raises(TypeError):
        UDTW(use_cuda=False)
    with pytest.raises(TypeError):
        torchwarp.SoftDTW(use_cuda=False)


def test_cost_on_sequence_pairs():
    g = torch.Generator().manual_seed(0)
    X, Y = torch.rand(2, 5, 3, generator=g, **f64), torch.rand(2, 7, 3, generator=g, **f64)
    diff = X.unsqueeze(2) - Y.unsqueeze(1)                         # [2, 5, 7, 3]
    torch.testing.assert_close(torchwarp.squared_euclidean_cost(X, Y), diff.pow(2).sum(-1))
    torch.testing.assert_close(torchwarp.euclidean_cost(X, Y), diff.pow(2).sum(-1).sqrt())
    torch.testing.assert_close(torchwarp.rbf_cost(X, Y, sigma=0.5),
                               (2 - 2 * torch.exp(-0.5 * diff.pow(2))).sum(-1))


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_udtw_module(backend, device):
    g = torch.Generator().manual_seed(0)
    X, Y = torch.rand(3, 6, 4, generator=g, **f64), torch.rand(3, 8, 4, generator=g, **f64)
    SX, SY = torch.rand(3, 6, 1, generator=g, **f64) + 0.5, torch.rand(3, 8, 1, generator=g, **f64) + 0.5
    xs = [t.to(device).requires_grad_() for t in (X, Y, SX, SY)]
    d, p = UDTW(gamma=0.3, beta=0.5, backend=backend)(*xs)
    assert d.shape == p.shape == (3,)
    (d.sum() + p.sum()).backward()
    assert all(torch.isfinite(x.grad).all() for x in xs)
    # the two distance formulations agree
    d2, p2 = UDTW(gamma=0.3, beta=0.5, backend=backend, method="diff")(*xs)
    torch.testing.assert_close(d, d2)
    torch.testing.assert_close(p, p2)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_jeanie_module(backend, device):
    g = torch.Generator().manual_seed(1)
    s = torch.randn(2, 7, 5, generator=g, **f64).to(device)
    q1 = torch.randn(2, 4, 6, 5, generator=g, **f64).to(device)        # one viewpoint axis
    q2 = torch.randn(2, 3, 2, 6, 5, generator=g, **f64).to(device)     # two viewpoint axes

    d1, R1 = JEANIE(gamma=0.1, max_shift=1, backend=backend)(q1, s, return_accumulator=True)
    assert d1.shape == (2,) and R1.shape == (2, 4, 6, 7)
    torch.testing.assert_close(d1, TF.jeanie(q1, s, gamma=0.1, max_shift=1, backend=backend))

    d2 = JEANIE(gamma=0.1, max_shift=(1, 1), backend=backend)(q2, s)
    torch.testing.assert_close(d2, TF.jeanie(q2, s, gamma=0.1, max_shift=(1, 1), backend=backend))
    assert "max_shift=(1, 1)" in repr(JEANIE(gamma=0.1, max_shift=(1, 1)))
    # an int max_shift applies to both viewpoint axes of a 5-D query
    torch.testing.assert_close(JEANIE(gamma=0.1, max_shift=1, backend=backend)(q2, s), d2)
    d20 = JEANIE(gamma=0.1, max_shift=(1, 0), backend=backend)(q2, s)
    torch.testing.assert_close(
        d20, TF.jeanie_from_cost(torchwarp.euclidean_cost(q2, s.unsqueeze(1)), gamma=0.1,
                                 max_shift=(1, 0), backend=backend))
    # a max_shift tuple must have one entry per viewpoint axis
    with pytest.raises(ValueError):
        JEANIE(gamma=0.1, max_shift=(1, 1))(q1, s)
    with pytest.raises(ValueError):
        JEANIE(gamma=0.1, max_shift=(1,))(q2, s)
    with pytest.raises(ValueError):
        TF.jeanie(q2, s, max_shift=(1, 1, 1))
    with pytest.raises(ValueError):
        TF.jeanie_from_cost(torch.rand(2, 3, 4, 5, **f64).to(device), max_shift=(1, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_backends_agree():
    g = torch.Generator().manual_seed(2)
    cost = torch.rand(4, 3, 2, 7, 9, generator=g, **f64)
    outs = [_jeanie_dp(cost.to(d), 0.2, 1, 1, b)[0].cpu() for b, d in BACKENDS]
    for o in outs[1:]:
        torch.testing.assert_close(o, outs[0], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_gradcheck(backend, device):
    kw = dict(device=device, **f64)
    c = torch.rand(2, 3, 4, **kw).requires_grad_()
    q = torch.randn(2, 3, 4, **kw).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda a, b: TF.udtw_from_cost(a, b, gamma=0.4, backend=backend), (c, q))
    cost = torch.rand(2, 2, 1, 3, 4, **kw).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda a: _jeanie_dp(a, 0.4, 1, 0, backend)[0], (cost,))


def _naive_soft_dtw(D, gamma, band=0):
    """soft-DTW recursion, with the Sakoe-Chiba band of pytorch-softdtw-cuda."""
    N, M = D.shape
    R = torch.full((N + 1, M + 1), float("inf"), **f64)
    R[0, 0] = 0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if 0 < band < abs(i - j):
                continue
            r = torch.stack([R[i - 1, j - 1], R[i - 1, j], R[i, j - 1]])
            R[i, j] = D[i - 1, j - 1] - gamma * torch.logsumexp(-r / gamma, 0)
    return R[N, M]


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("bandwidth", [None, 2])
def test_soft_dtw_module(backend, device, bandwidth):
    g = torch.Generator().manual_seed(3)
    X = torch.randn(3, 6, 2, generator=g, **f64).to(device).requires_grad_()
    Y = torch.randn(3, 8, 2, generator=g, **f64).to(device)
    sdtw = torchwarp.SoftDTW(gamma=0.5, bandwidth=bandwidth, backend=backend)
    d = sdtw(X, Y)
    D = torchwarp.squared_euclidean_cost(X, Y).detach().cpu()
    want = torch.stack([_naive_soft_dtw(D[b], 0.5, bandwidth or 0) for b in range(3)])
    torch.testing.assert_close(d.detach().cpu(), want)
    d.sum().backward()
    assert torch.isfinite(X.grad).all()

    div = torchwarp.SoftDTW(gamma=0.5, normalize=True, bandwidth=bandwidth, backend=backend)
    torch.testing.assert_close(div(X, Y), d - 0.5 * (sdtw(X, X) + sdtw(Y, Y)))
    l1 = lambda a, b: (a.unsqueeze(2) - b.unsqueeze(1)).abs().sum(-1)
    d1 = torchwarp.SoftDTW(gamma=0.5, dist_func=l1, backend=backend)(X, Y)
    torch.testing.assert_close(d1, TF.soft_dtw_from_cost(l1(X, Y), gamma=0.5))
    # the functional form and soft_dtw_from_cost on the default distance agree
    torch.testing.assert_close(TF.soft_dtw(X, Y, gamma=0.5, bandwidth=bandwidth, backend=backend),
                               d)
    torch.testing.assert_close(
        TF.soft_dtw_from_cost(torchwarp.squared_euclidean_cost(X, Y), gamma=0.5,
                              bandwidth=bandwidth, backend=backend), d)


def test_soft_dtw_rejects_unbatched():
    X, Y = torch.randn(6, 2, **f64), torch.randn(8, 2, **f64)
    with pytest.raises(ValueError):
        torchwarp.SoftDTW(gamma=0.5)(X, Y)
    with pytest.raises(ValueError):
        TF.soft_dtw_from_cost(torchwarp.squared_euclidean_cost(X, Y), gamma=0.5)


def test_softmin_weights():
    v = torch.randn(3, 4, **f64)
    torch.testing.assert_close(TF.softmin_weights(v, 0.5, dim=1),
                               torch.softmax(-v / 0.5, dim=1))
    torch.testing.assert_close(TF.softmin(v, 0.5, dim=1),
                               -0.5 * torch.logsumexp(-v / 0.5, dim=1))


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_udtw_functional_normalize_matches_module(backend, device):
    g = torch.Generator().manual_seed(5)
    X, Y = torch.randn(2, 5, 3, generator=g, **f64), torch.randn(2, 7, 3, generator=g, **f64)
    sx = torch.rand(2, 5, 1, generator=g, **f64) + 0.5
    sy = torch.rand(2, 7, 1, generator=g, **f64) + 0.5
    xs = [t.to(device) for t in (X, Y, sx, sy)]
    for method in ("gemm", "diff"):
        want = UDTW(gamma=0.3, beta=0.6, normalize=True, method=method, backend=backend)(*xs)
        got = TF.udtw(*xs, beta=0.6, gamma=0.3, normalize=True, method=method, backend=backend)
        for a, b in zip(got, want):
            torch.testing.assert_close(a, b)
        # normalize=True is d(X,Y) - 0.5 [d(X,X) + d(Y,Y)]
        d = lambda a, b, sa, sb: TF.udtw(a, b, sa, sb, beta=0.6, gamma=0.3, method=method,
                                         backend=backend)
        xy, xx, yy = d(*xs), d(xs[0], xs[0], xs[2], xs[2]), d(xs[1], xs[1], xs[3], xs[3])
        for i in range(2):
            torch.testing.assert_close(got[i], xy[i] - 0.5 * (xx[i] + yy[i]))


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "rbf"])
def test_fvm_module(backend, device, metric):
    g = torch.Generator().manual_seed(6)
    s = torch.randn(2, 7, 5, generator=g, **f64).to(device)
    q1 = torch.randn(2, 4, 6, 5, generator=g, **f64).to(device)        # one viewpoint axis
    q2 = torch.randn(2, 3, 2, 6, 5, generator=g, **f64).to(device)     # two viewpoint axes
    cost_fn = {"euclidean": torchwarp.euclidean_cost,
               "sqeuclidean": torchwarp.squared_euclidean_cost, "rbf": torchwarp.rbf_cost}[metric]
    m = FVM(gamma=0.2, metric=metric, backend=backend)
    torch.testing.assert_close(
        m(q1, s), TF.fvm_from_cost(cost_fn(q1, s), gamma=0.2, backend=backend))
    torch.testing.assert_close(
        m(q2, s), TF.fvm_from_cost(cost_fn(q2, s.unsqueeze(1)), gamma=0.2, backend=backend))
    torch.testing.assert_close(m(q1, s), TF.fvm(q1, s, gamma=0.2, metric=metric, backend=backend))
    assert "gamma=0.2" in repr(m)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_fvm_support_viewpoints(backend, device):
    g = torch.Generator().manual_seed(7)
    q = torch.randn(2, 4, 6, 5, generator=g, **f64).to(device)    # [B, Kq, T, D]
    s = torch.randn(2, 3, 7, 5, generator=g, **f64).to(device)    # [B, Ks, U, D]
    cost = torchwarp.euclidean_cost(q.unsqueeze(2), s.unsqueeze(1))  # [B, Kq, Ks, T, U]
    assert cost.shape == (2, 4, 3, 6, 7)
    want = TF.fvm_from_cost(cost, gamma=0.1, backend=backend)
    torch.testing.assert_close(TF.fvm(q, s, gamma=0.1, backend=backend), want)
    torch.testing.assert_close(FVM(gamma=0.1, backend=backend)(q, s), want)
    # one SoftMin over all viewpoint pairs: flattening the two dims gives the same
    flat = cost.reshape(2, 12, 6, 7)
    torch.testing.assert_close(TF.fvm_from_cost(flat, gamma=0.1, backend=backend), want)


def test_udtw_output_defaults_keyword_only():
    g = torch.Generator().manual_seed(4)
    X, Y = torch.randn(2, 5, 3, generator=g, **f64), torch.randn(2, 7, 3, generator=g, **f64)
    sx = torch.rand(2, 5, 1, generator=g, **f64) + 0.5
    sy = torch.rand(2, 7, 1, generator=g, **f64) + 0.5
    out = UDTW()(X, Y, sx, sy)
    d, omega = out
    assert out.distance is d and out.penalty is omega
    torch.testing.assert_close(TF.udtw(X, Y, sx, sy).distance, d)
    # torchwarp.paths has the defaults of the computation
    torch.testing.assert_close(torchwarp.paths.udtw(X, Y, sx, sy).distance, d)
    # options of torchwarp's own functions are keyword-only
    with pytest.raises(TypeError):
        TF.udtw(X, Y, sx, sy, 0.1, 1.0)
    with pytest.raises(TypeError):
        TF.udtw_from_cost(X @ Y.transpose(1, 2), X @ Y.transpose(1, 2), 0.1)
    with pytest.raises(TypeError):
        TF.fvm_from_cost(torch.rand(1, 2, 3, 4), 0.1)
    with pytest.raises(TypeError):
        TF.soft_dtw_from_cost(torch.rand(1, 3, 4), 0.1)
    # a batch dimension is required
    with pytest.raises(ValueError):
        TF.udtw(X[0], Y[0], sx[0], sy[0])
    with pytest.raises(ValueError):
        TF.udtw_from_cost(torch.rand(3, 4), torch.rand(3, 4))
    with pytest.raises(TypeError):
        torchwarp.paths.udtw(X, Y, sx, sy, 0.1)
