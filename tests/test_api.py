"""Public API checks that do not need the official implementation."""

import pytest
import torch

import torchwarp
from torchwarp import JEANIE, _cuda, uDTW

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


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_udtw_module(backend, device):
    g = torch.Generator().manual_seed(0)
    X, Y = torch.rand(3, 6, 4, generator=g, **f64), torch.rand(3, 8, 4, generator=g, **f64)
    SX, SY = 0.5 + torch.rand(3, 6, 1, generator=g, **f64), 0.5 + torch.rand(3, 8, 1, generator=g, **f64)
    xs = [t.to(device).requires_grad_() for t in (X, Y, SX, SY)]
    d, p = uDTW(gamma=0.3, backend=backend)(*xs, beta=0.5)
    assert d.shape == p.shape == (3,)
    (d.sum() + p.sum()).backward()
    assert all(torch.isfinite(x.grad).all() for x in xs)
    # the two distance formulations agree
    d2, p2 = uDTW(gamma=0.3, backend=backend, distance="diff")(*xs, beta=0.5)
    torch.testing.assert_close(d, d2)
    torch.testing.assert_close(p, p2)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_jeanie_module(backend, device):
    g = torch.Generator().manual_seed(1)
    s = torch.randn(2, 7, 5, generator=g, **f64).to(device)
    q1 = torch.randn(2, 4, 6, 5, generator=g, **f64).to(device)        # one viewpoint axis
    q2 = torch.randn(2, 3, 2, 6, 5, generator=g, **f64).to(device)     # two viewpoint axes

    d1, R1 = JEANIE(0.1, 1, backend=backend)(q1, s, return_accumulator=True)
    assert d1.shape == (2,) and R1.shape == (2, 4, 6, 7)
    torch.testing.assert_close(
        d1, torchwarp.jeanie_1d_from_features(q1, s, 0.1, 1, backend=backend))

    d2 = JEANIE(0.1, (1, 1), backend=backend)(q2, s)
    torch.testing.assert_close(
        d2, torchwarp.jeanie_2d_from_features(q2, s, 0.1, 1, 1, backend=backend))
    assert "max_shift=(1, 1)" in repr(JEANIE(0.1, (1, 1)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_backends_agree():
    g = torch.Generator().manual_seed(2)
    cost = torch.rand(4, 3, 2, 7, 9, generator=g, **f64)
    outs = [torchwarp.jeanie_dp(cost.to(d), 0.2, 1, 1, b)[0].cpu() for b, d in BACKENDS]
    for o in outs[1:]:
        torch.testing.assert_close(o, outs[0], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_gradcheck(backend, device):
    kw = dict(device=device, **f64)
    c = torch.rand(2, 3, 4, **kw).requires_grad_()
    q = torch.randn(2, 3, 4, **kw).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda a, b: torchwarp.udtw_from_matrices(a, b, gamma=0.4, backend=backend), (c, q))
    cost = torch.rand(2, 2, 1, 3, 4, **kw).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda a: torchwarp.jeanie_dp(a, 0.4, 1, 0, backend)[0], (cost,))


def test_testing_reference_names():
    from torchwarp.testing import _UPSTREAM
    assert set(_UPSTREAM) == {"udtw", "jeanie"}
