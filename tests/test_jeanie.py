"""torchwarp JEANIE / soft-DTW / FVM vs the upstream reference (float64)."""

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

ATOL = 1e-10


def _value_and_grad(fn, cost, device):
    c = cost.detach().to(device).requires_grad_(True)
    out = fn(c)
    out.sum().backward()
    return out.detach().cpu(), c.grad.cpu()


def _assert_close(a, b):
    torch.testing.assert_close(a, b, atol=ATOL, rtol=1e-9)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("T,U", [(1, 1), (1, 6), (6, 1), (5, 7), (8, 4), (9, 9)])
@pytest.mark.parametrize("gamma", [0.01, 0.1, 1.0])
def test_soft_dtw(backend, device, T, U, gamma):
    cost = torch.rand(T, U, dtype=torch.float64)
    v_ref, g_ref = _value_and_grad(lambda c: ref.soft_dtw(c, gamma), cost, "cpu")
    v, g = _value_and_grad(lambda c: fast.soft_dtw(c, gamma, backend), cost, device)
    _assert_close(v, v_ref)
    _assert_close(g, g_ref)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("K,T,U", [(1, 4, 5), (5, 6, 7), (5, 8, 5), (3, 1, 4)])
@pytest.mark.parametrize("shift", [0, 1, 2, 7])
def test_jeanie_1d(backend, device, K, T, U, shift):
    cost = torch.rand(K, T, U, dtype=torch.float64)
    gamma = 0.1
    v_ref, g_ref = _value_and_grad(
        lambda c: ref.jeanie_1d_from_cost(c, gamma, shift), cost, "cpu")
    v, g = _value_and_grad(
        lambda c: fast.jeanie_1d_from_cost(c, gamma, shift, backend=backend), cost, device)
    _assert_close(v, v_ref)
    _assert_close(g, g_ref)

    _, acc_ref = ref.jeanie_1d_from_cost(cost, gamma, shift, return_accumulator=True)
    _, acc = fast.jeanie_1d_from_cost(
        cost.to(device), gamma, shift, return_accumulator=True, backend=backend)
    _assert_close(acc.cpu(), acc_ref.detach())


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("K1,K2,T,U", [(3, 3, 5, 6), (2, 4, 6, 4), (1, 3, 4, 4)])
@pytest.mark.parametrize("s_az,s_alt", [(0, 0), (1, 1), (1, 0), (2, 1)])
def test_jeanie_2d(backend, device, K1, K2, T, U, s_az, s_alt):
    cost = torch.rand(K1, K2, T, U, dtype=torch.float64)
    gamma = 0.1
    v_ref, g_ref = _value_and_grad(
        lambda c: ref.jeanie_2d_from_cost(c, gamma, s_az, s_alt), cost, "cpu")
    v, g = _value_and_grad(
        lambda c: fast.jeanie_2d_from_cost(c, gamma, s_az, s_alt, backend=backend),
        cost, device)
    _assert_close(v, v_ref)
    _assert_close(g, g_ref)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_batched_matches_loop(backend, device):
    cost = torch.rand(6, 4, 7, 9, dtype=torch.float64)
    batched = fast.jeanie_1d_from_cost(cost.to(device), 0.1, 1, backend=backend).cpu()
    looped = torch.stack([ref.jeanie_1d_from_cost(c, 0.1, 1) for c in cost])
    _assert_close(batched, looped.detach())

    cost2 = torch.rand(3, 3, 3, 5, 6, dtype=torch.float64)
    batched = fast.jeanie_2d_from_cost(cost2.to(device), 0.1, 1, 1, backend=backend).cpu()
    looped = torch.stack([ref.jeanie_2d_from_cost(c, 0.1, 1, 1) for c in cost2])
    _assert_close(batched, looped.detach())


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("name,shape", [
    ("fvm_query_only_1d", (4, 5, 6)),
    ("fvm_query_only_2d", (3, 3, 5, 6)),
    ("fvm_1d_from_cost", (3, 4, 6, 5)),
    ("fvm_2d_from_cost", (2, 2, 3, 2, 5, 6)),
])
def test_fvm(backend, device, name, shape):
    cost = torch.rand(*shape, dtype=torch.float64)
    v_ref, g_ref = _value_and_grad(lambda c: getattr(ref, name)(c, 0.1), cost, "cpu")
    v, g = _value_and_grad(
        lambda c: getattr(fast, name)(c, 0.1, backend=backend), cost, device)
    _assert_close(v, v_ref)
    _assert_close(g, g_ref)

    batch = torch.rand(3, *shape, dtype=torch.float64)
    looped = torch.stack([getattr(ref, name)(c, 0.1) for c in batch])
    _assert_close(getattr(fast, name)(batch.to(device), 0.1, backend=backend).cpu(), looped)


@pytest.mark.parametrize("name", ["euclidean_cost", "squared_euclidean_cost", "rbf_cost"])
def test_distances(name):
    q = torch.randn(4, 5, 8, dtype=torch.float64, requires_grad=True)
    s = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
    out_ref = getattr(ref, name)(q, s)
    g_ref = torch.autograd.grad(out_ref.sum(), (q, s))
    out = getattr(fast, name)(q, s)
    g = torch.autograd.grad(out.sum(), (q, s))
    _assert_close(out, out_ref)
    for a, b in zip(g, g_ref):
        _assert_close(a, b)

    qb = torch.randn(3, 4, 5, 8, dtype=torch.float64)
    sb = torch.randn(3, 6, 8, dtype=torch.float64)
    looped = torch.stack([getattr(ref, name)(a, b) for a, b in zip(qb, sb)])
    _assert_close(getattr(fast, name)(qb, sb), looped)


def test_euclidean_zero_distance_grad_is_finite():
    q = torch.ones(1, 2, 3, dtype=torch.float64, requires_grad=True)
    s = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    fast.euclidean_cost(q, s).sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(s.grad).all()


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_gradcheck(backend, device):
    cost = torch.rand(2, 3, 2, 4, 5, dtype=torch.float64, device=device, requires_grad=True)
    fn = lambda c: fast.jeanie_dp(c, 0.5, 1, 1, backend)[0]
    assert torch.autograd.gradcheck(fn, (cost,))


def test_float32_accuracy():
    # float32 vs the float64 truth, as max-normalised error.
    cost = torch.rand(8, 5, 12, 15, dtype=torch.float64)
    v_ref, g_ref = _value_and_grad(
        lambda c: fast.jeanie_1d_from_cost(c, 0.1, 1, backend="torch"), cost, "cpu")
    for backend, device in BACKENDS:
        v, g = _value_and_grad(
            lambda c: fast.jeanie_1d_from_cost(c, 0.1, 1, backend=backend),
            cost.float(), device)
        for a, b in ((v, v_ref), (g, g_ref)):
            err = ((a.double() - b).abs().max() / b.abs().max()).item()
            assert err < 1e-4, (backend, device, err)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA extension unavailable")
def test_large_problem_uses_global_scratch():
    # 3 * T * K * 8 bytes > 48 KiB forces the global-memory buffer path.
    cost = torch.rand(2, 25, 1, 90, 95, dtype=torch.float64)
    a = fast.jeanie_dp(cost, 0.1, 2, 0, "torch")[0]
    b = fast.jeanie_dp(cost.cuda(), 0.1, 2, 0, "cuda")[0].cpu()
    _assert_close(b, a)


def test_end_to_end_feature_gradients():
    q = torch.randn(5, 6, 16, dtype=torch.float64, requires_grad=True)
    s = torch.randn(7, 16, dtype=torch.float64, requires_grad=True)
    d_ref = ref.jeanie_1d_from_cost(ref.euclidean_cost(q, s), 0.1, 1)
    g_ref = torch.autograd.grad(d_ref, (q, s))
    d = fast.jeanie_1d_from_cost(fast.euclidean_cost(q, s), 0.1, 1)
    g = torch.autograd.grad(d, (q, s))
    _assert_close(d, d_ref)
    for a, b in zip(g, g_ref):
        _assert_close(a, b)


_REF_COST = {"euclidean": ref.euclidean_cost, "sqeuclidean": ref.squared_euclidean_cost,
             "rbf": ref.rbf_cost}


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "rbf"])
@pytest.mark.parametrize("K,T,U,shift", [(5, 6, 8, 1), (3, 9, 4, 2), (1, 5, 5, 0)])
def test_jeanie_1d_from_features(backend, device, metric, K, T, U, shift):
    q = torch.randn(2, K, T, 6, dtype=torch.float64)
    s = torch.randn(2, U, 6, dtype=torch.float64)

    def reference(q, s):
        return torch.stack([ref.jeanie_1d_from_cost(_REF_COST[metric](a, b), 0.1, shift)
                            for a, b in zip(q, s)])

    qa, sa = q.clone().requires_grad_(), s.clone().requires_grad_()
    d_ref = reference(qa, sa)
    g_ref = torch.autograd.grad(d_ref.sum(), (qa, sa))

    qb, sb = q.to(device).requires_grad_(), s.to(device).requires_grad_()
    d = fast.jeanie_1d_from_features(qb, sb, 0.1, shift, metric=metric, backend=backend)
    g = torch.autograd.grad(d.sum(), (qb, sb))
    _assert_close(d.detach().cpu(), d_ref.detach())
    for a, b in zip(g, g_ref):
        _assert_close(a.cpu(), b)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean"])
def test_jeanie_2d_from_features(backend, device, metric):
    q = torch.randn(3, 3, 2, 5, 4, dtype=torch.float64)
    s = torch.randn(3, 7, 4, dtype=torch.float64)

    def reference(q, s):
        return torch.stack([
            ref.jeanie_2d_from_cost(
                _REF_COST[metric](a.reshape(6, 5, 4), b).reshape(3, 2, 5, 7), 0.1, 1, 1)
            for a, b in zip(q, s)])

    qa, sa = q.clone().requires_grad_(), s.clone().requires_grad_()
    d_ref = reference(qa, sa)
    g_ref = torch.autograd.grad(d_ref.sum(), (qa, sa))
    qb, sb = q.to(device).requires_grad_(), s.to(device).requires_grad_()
    d, acc = fast.jeanie_2d_from_features(qb, sb, 0.1, 1, 1, metric=metric,
                                          return_accumulator=True, backend=backend)
    g = torch.autograd.grad(d.sum(), (qb, sb))
    _assert_close(d.detach().cpu(), d_ref.detach())
    for a, b in zip(g, g_ref):
        _assert_close(a.cpu(), b)
    _, acc_ref = ref.jeanie_2d_from_cost(
        _REF_COST[metric](q[0].reshape(6, 5, 4), s[0]).reshape(3, 2, 5, 7), 0.1, 1, 1,
        return_accumulator=True)
    _assert_close(acc[0].cpu(), acc_ref.detach())


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA extension unavailable")
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean"])
def test_fused_features_gradcheck(metric):
    q = torch.randn(2, 2, 2, 4, 3, dtype=torch.float64, device="cuda", requires_grad=True)
    s = torch.randn(2, 5, 3, dtype=torch.float64, device="cuda", requires_grad=True)
    fn = lambda q, s: fast.jeanie_dp_features(q, s, 0.5, 1, 1, metric, "cuda")[0]
    assert torch.autograd.gradcheck(fn, (q, s))


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA extension unavailable")
def test_fused_features_large_uses_global_scratch():
    q = torch.randn(2, 25, 1, 90, 8, dtype=torch.float64)
    s = torch.randn(2, 95, 8, dtype=torch.float64)
    a = fast.jeanie_dp_features(q, s, 0.1, 2, 0, "euclidean", "torch")[0]
    b = fast.jeanie_dp_features(q.cuda(), s.cuda(), 0.1, 2, 0, "euclidean", "cuda")[0]
    _assert_close(b.cpu(), a)


@pytest.mark.parametrize("backend,device", BACKENDS, ids=IDS)
def test_features_zero_distance_grad_is_finite(backend, device):
    q = torch.ones(1, 2, 3, 4, dtype=torch.float64, device=device, requires_grad=True)
    s = torch.ones(1, 3, 4, dtype=torch.float64, device=device, requires_grad=True)
    fast.jeanie_1d_from_features(q, s, 0.1, 1, backend=backend).sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(s.grad).all()


def test_input_validation():
    with pytest.raises(ValueError):
        fast.jeanie_1d_from_cost(torch.rand(4, 5), 0.1)
    with pytest.raises(ValueError):
        fast.jeanie_1d_from_cost(torch.rand(2, 4, 5), 0.0)
    with pytest.raises(ValueError):
        fast.jeanie_1d_from_cost(torch.rand(2, 4, 5), 0.1, -1)
    with pytest.raises(TypeError):
        fast.soft_dtw(torch.ones(3, 3, dtype=torch.int64))
