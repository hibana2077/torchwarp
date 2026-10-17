"""torchwarp JEANIE / soft-DTW / FVM vs the stored outputs of the official implementation (float64)."""

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

ATOL = 1e-10


def _value_and_grad(fn, cost, device):
    c = cost.detach().to(device).requires_grad_(True)
    out = fn(c)
    out.sum().backward()
    return out.detach().cpu(), c.grad.cpu()


def _assert_close(a, b):
    torch.testing.assert_close(a, b, atol=ATOL, rtol=1e-9)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("T,U", G.SDTW_SHAPES)
@pytest.mark.parametrize("gamma", G.GAMMAS)
def test_soft_dtw(backend, device, T, U, gamma):
    ref = golden("jeanie.soft_dtw", T=T, U=U, gamma=gamma)
    v, g = _value_and_grad(
        lambda c: TF.soft_dtw_from_cost(c[None], gamma=gamma, backend=backend)[0],
        ref["cost"], device)
    _assert_close(v, ref["value"])
    _assert_close(g, ref["grad"])


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("K,T,U", G.JEANIE_1D_SHAPES)
@pytest.mark.parametrize("shift", G.JEANIE_1D_SHIFTS)
def test_jeanie_1d(backend, device, K, T, U, shift):
    ref = golden("jeanie.jeanie_1d", K=K, T=T, U=U, shift=shift)
    cost = ref["cost"]
    v, g = _value_and_grad(
        lambda c: TF.jeanie_from_cost(c[None], gamma=0.1, max_shift=shift, backend=backend)[0],
        cost, device)
    _assert_close(v, ref["value"])
    _assert_close(g, ref["grad"])

    _, acc = TF.jeanie_from_cost(cost[None].to(device), gamma=0.1, max_shift=shift,
                                 return_accumulator=True, backend=backend)
    _assert_close(acc[0].cpu(), ref["accumulator"])


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("K1,K2,T,U", G.JEANIE_2D_SHAPES)
@pytest.mark.parametrize("s_az,s_alt", G.JEANIE_2D_SHIFTS)
def test_jeanie_2d(backend, device, K1, K2, T, U, s_az, s_alt):
    ref = golden("jeanie.jeanie_2d", K1=K1, K2=K2, T=T, U=U, s_az=s_az, s_alt=s_alt)
    v, g = _value_and_grad(
        lambda c: TF.jeanie_from_cost(c[None], gamma=0.1, max_shift=(s_az, s_alt),
                                      backend=backend)[0],
        ref["cost"], device)
    _assert_close(v, ref["value"])
    _assert_close(g, ref["grad"])


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_batched_matches_loop(backend, device):
    ref = golden("jeanie.batched")
    batched = TF.jeanie_from_cost(ref["cost_1d"].to(device), gamma=0.1, max_shift=1,
                                  backend=backend).cpu()
    _assert_close(batched, ref["value_1d"])

    batched = TF.jeanie_from_cost(ref["cost_2d"].to(device), gamma=0.1, max_shift=(1, 1),
                                  backend=backend).cpu()
    _assert_close(batched, ref["value_2d"])


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
# every official FVM variant is TF.fvm_from_cost on the batched cost
@pytest.mark.parametrize("name,shape", G.FVM_CASES)
def test_fvm(backend, device, name, shape):
    ref = golden("jeanie.fvm", name=name)
    v, g = _value_and_grad(
        lambda c: TF.fvm_from_cost(c[None], gamma=0.1, backend=backend)[0], ref["cost"], device)
    _assert_close(v, ref["value"])
    _assert_close(g, ref["grad"])
    _assert_close(TF.fvm_from_cost(ref["batch"].to(device), gamma=0.1, backend=backend).cpu(),
                  ref["batch_value"])


def test_softmin_matches_reference():
    ref = golden("jeanie.softmin")
    torch.testing.assert_close(TF.softmin(ref["v"], 0.3, dim=1), ref["outputs"][0])
    torch.testing.assert_close(TF.softmin(ref["v"], 0.3), ref["outputs"][1])


@pytest.mark.parametrize("name", G.DISTANCES)
def test_distances(name):
    ref = golden("jeanie.distance", name=name)
    q, s = ref["q"].clone().requires_grad_(), ref["s"].clone().requires_grad_()
    out = getattr(fast, name)(q, s)
    g = torch.autograd.grad(out.sum(), (q, s))
    _assert_close(out, ref["value"])
    for a, b in zip(g, ref["grads"]):
        _assert_close(a, b)
    _assert_close(getattr(fast, name)(ref["qb"], ref["sb"]), ref["batch_value"])


def test_euclidean_zero_distance_grad_is_finite():
    q = torch.ones(1, 2, 3, dtype=torch.float64, requires_grad=True)
    s = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    fast.euclidean_cost(q, s).sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(s.grad).all()


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_gradcheck(backend, device):
    cost = torch.rand(2, 3, 2, 4, 5, dtype=torch.float64, device=device, requires_grad=True)
    fn = lambda c: _jeanie_dp(c, 0.5, 1, 1, backend)[0]
    assert torch.autograd.gradcheck(fn, (cost,))


def test_float32_accuracy():
    # float32 vs the float64 truth, as max-normalised error.
    cost = torch.rand(8, 5, 12, 15, dtype=torch.float64)
    v_ref, g_ref = _value_and_grad(
        lambda c: TF.jeanie_from_cost(c, gamma=0.1, max_shift=1, backend="torch"), cost, "cpu")
    for backend, device in BACKENDS:
        v, g = _value_and_grad(
            lambda c: TF.jeanie_from_cost(c, gamma=0.1, max_shift=1, backend=backend),
            cost.float(), device)
        for a, b in ((v, v_ref), (g, g_ref)):
            err = ((a.double() - b).abs().max() / b.abs().max()).item()
            assert err < 1e-4, (backend, device, err)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA extension unavailable")
def test_large_problem_uses_global_scratch():
    # 3 * T * K * 8 bytes > 48 KiB forces the global-memory buffer path.
    cost = torch.rand(2, 25, 1, 90, 95, dtype=torch.float64)
    a = _jeanie_dp(cost, 0.1, 2, 0, "torch")[0]
    b = _jeanie_dp(cost.cuda(), 0.1, 2, 0, "cuda")[0].cpu()
    _assert_close(b, a)


def test_end_to_end_feature_gradients():
    ref = golden("jeanie.end_to_end")
    q, s = ref["q"].clone().requires_grad_(), ref["s"].clone().requires_grad_()
    d = TF.jeanie_from_cost(fast.euclidean_cost(q[None], s[None]), gamma=0.1, max_shift=1)[0]
    g = torch.autograd.grad(d, (q, s))
    _assert_close(d, ref["value"])
    for a, b in zip(g, ref["grads"]):
        _assert_close(a, b)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("metric", G.METRICS)
@pytest.mark.parametrize("K,T,U,shift", G.FEATURES_1D)
def test_jeanie_features_1d(backend, device, metric, K, T, U, shift):
    ref = golden("jeanie.features_1d", metric=metric, K=K, T=T, U=U, shift=shift)
    qb, sb = ref["q"].to(device).requires_grad_(), ref["s"].to(device).requires_grad_()
    d = TF.jeanie(qb, sb, gamma=0.1, max_shift=shift, metric=metric, backend=backend)
    g = torch.autograd.grad(d.sum(), (qb, sb))
    _assert_close(d.detach().cpu(), ref["value"])
    for a, b in zip(g, ref["grads"]):
        _assert_close(a.cpu(), b)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean"])
def test_jeanie_features_2d(backend, device, metric):
    ref = golden("jeanie.features_2d", metric=metric)
    qb, sb = ref["q"].to(device).requires_grad_(), ref["s"].to(device).requires_grad_()
    d, acc = TF.jeanie(qb, sb, gamma=0.1, max_shift=(1, 1), metric=metric,
                       return_accumulator=True, backend=backend)
    g = torch.autograd.grad(d.sum(), (qb, sb))
    _assert_close(d.detach().cpu(), ref["value"])
    for a, b in zip(g, ref["grads"]):
        _assert_close(a.cpu(), b)
    _assert_close(acc[0].detach().cpu(), ref["accumulator"])


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA extension unavailable")
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean"])
def test_fused_features_gradcheck(metric):
    q = torch.randn(2, 2, 2, 4, 3, dtype=torch.float64, device="cuda", requires_grad=True)
    s = torch.randn(2, 5, 3, dtype=torch.float64, device="cuda", requires_grad=True)
    fn = lambda q, s: _jeanie_dp_features(q, s, 0.5, 1, 1, metric, "cuda")[0]
    assert torch.autograd.gradcheck(fn, (q, s))


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA extension unavailable")
def test_fused_features_large_uses_global_scratch():
    q = torch.randn(2, 25, 1, 90, 8, dtype=torch.float64)
    s = torch.randn(2, 95, 8, dtype=torch.float64)
    a = _jeanie_dp_features(q, s, 0.1, 2, 0, "euclidean", "torch")[0]
    b = _jeanie_dp_features(q.cuda(), s.cuda(), 0.1, 2, 0, "euclidean", "cuda")[0]
    _assert_close(b.cpu(), a)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_features_zero_distance_grad_is_finite(backend, device):
    q = torch.ones(1, 2, 3, 4, dtype=torch.float64, device=device, requires_grad=True)
    s = torch.ones(1, 3, 4, dtype=torch.float64, device=device, requires_grad=True)
    TF.jeanie(q, s, gamma=0.1, max_shift=1, backend=backend).sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(s.grad).all()


def test_input_validation():
    with pytest.raises(ValueError):
        TF.jeanie_from_cost(torch.rand(4, 5), gamma=0.1)
    with pytest.raises(ValueError):                 # unbatched [K, T, U] is not accepted
        TF.jeanie_from_cost(torch.rand(2, 4, 5), gamma=0.1)
    with pytest.raises(ValueError):
        TF.jeanie_from_cost(torch.rand(1, 2, 4, 5), gamma=0.0)
    with pytest.raises(ValueError):
        TF.jeanie_from_cost(torch.rand(1, 2, 4, 5), gamma=0.1, max_shift=-1)
    with pytest.raises(ValueError):                 # unbatched [K, T, D] / [U, D]
        TF.jeanie(torch.rand(2, 4, 3), torch.rand(5, 3))
    with pytest.raises(TypeError):
        TF.soft_dtw_from_cost(torch.ones(1, 3, 3, dtype=torch.int64))
    with pytest.raises(ValueError):                 # unbatched [T, U]
        TF.soft_dtw_from_cost(torch.rand(3, 3))
    with pytest.raises(ValueError):                 # unbatched [K, T, U] is one viewpoint short
        TF.fvm_from_cost(torch.rand(4, 5, 6))
    with pytest.raises(TypeError):                  # hyperparameters are keyword-only
        TF.jeanie_from_cost(torch.rand(1, 2, 4, 5), 0.1)
