"""Public API checks that do not need the official implementation."""

import pytest
import torch

import torchwarp
from torchwarp import JEANIE, UDTW, _cuda
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
    assert torchwarp.__version__
    assert torchwarp.uDTW is torchwarp.UDTW


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
    d, p = UDTW(gamma=0.3, backend=backend)(*xs, beta=0.5)
    assert d.shape == p.shape == (3,)
    (d.sum() + p.sum()).backward()
    assert all(torch.isfinite(x.grad).all() for x in xs)
    # the two distance formulations agree
    d2, p2 = UDTW(gamma=0.3, backend=backend, method="diff")(*xs, beta=0.5)
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
    torch.testing.assert_close(
        d1, torchwarp.jeanie_1d_from_features(q1, s, gamma=0.1, max_shift=1, backend=backend))

    d2 = JEANIE(gamma=0.1, max_shift=(1, 1), backend=backend)(q2, s)
    torch.testing.assert_close(
        d2, torchwarp.jeanie_2d_from_features(q2, s, gamma=0.1, max_shift_az=1, max_shift_alt=1,
                                               backend=backend))
    assert "max_shift=(1, 1)" in repr(JEANIE(gamma=0.1, max_shift=(1, 1)))


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
        lambda a, b: torchwarp.udtw_from_cost(a, b, gamma=0.4, backend=backend), (c, q))
    cost = torch.rand(2, 2, 1, 3, 4, **kw).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda a: _jeanie_dp(a, 0.4, 1, 0, backend)[0], (cost,))


def test_reference_names():
    from torchwarp._reference import _UPSTREAM
    assert set(_UPSTREAM) == {"udtw", "jeanie"}


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
    torch.testing.assert_close(d1, torchwarp.soft_dtw(l1(X, Y), 0.5))


def test_softmin_weights():
    v = torch.randn(3, 4, **f64)
    torch.testing.assert_close(torchwarp.softmin_weights(v, 0.5, dim=1),
                               torch.softmax(-v / 0.5, dim=1))


def test_udtw_output_defaults_keyword_only():
    g = torch.Generator().manual_seed(4)
    X, Y = torch.randn(2, 5, 3, generator=g, **f64), torch.randn(2, 7, 3, generator=g, **f64)
    sx = torch.rand(2, 5, 1, generator=g, **f64) + 0.5
    sy = torch.rand(2, 7, 1, generator=g, **f64) + 0.5
    out = UDTW()(X, Y, sx, sy)
    d, omega = out
    assert out.distance is d and out.penalty is omega
    torch.testing.assert_close(torchwarp.udtw_from_features(X, Y, sx, sy).distance, d)
    # torchwarp.paths has the defaults of the computation
    torch.testing.assert_close(torchwarp.paths.udtw(X, Y, sx, sy).distance, d)
    # options of torchwarp's own functions are keyword-only
    with pytest.raises(TypeError):
        torchwarp.udtw_from_features(X, Y, sx, sy, 0.1, 1.0)
    with pytest.raises(TypeError):
        torchwarp.paths.udtw(X, Y, sx, sy, 0.1)
