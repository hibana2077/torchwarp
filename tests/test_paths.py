"""Alignment paths (torchwarp.paths) and the plotting helpers (torchwarp.plot)."""

import pytest
import torch

import torchwarp
from torchwarp import paths

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def exact_dtw(cost):
    T, U = cost.shape
    R = torch.full((T + 1, U + 1), float("inf"), dtype=cost.dtype)
    R[0, 0] = 0
    for t in range(1, T + 1):
        for u in range(1, U + 1):
            R[t, u] = cost[t - 1, u - 1] + min(R[t - 1, u], R[t, u - 1], R[t - 1, u - 1])
    return R[T, U]


def check_monotone(path, T, U):
    assert path[0][:2] == (0, 0) and path[-1][:2] == (T - 1, U - 1)
    for a, b in zip(path, path[1:]):
        assert (b[0] - a[0], b[1] - a[1]) in {(1, 0), (0, 1), (1, 1)}


@pytest.mark.parametrize("device", DEVICES)
def test_sdtw_hard_path_is_the_dtw_path(device):
    torch.manual_seed(0)
    cost = torch.rand(3, 9, 12, dtype=torch.float64, device=device)
    a = paths.sdtw(cost, gamma=1e-3)
    for b in range(3):
        check_monotone(a.path[b], 9, 12)
        along = sum(float(cost[b, t, u]) for t, u in a.path[b])
        assert along == pytest.approx(float(exact_dtw(cost[b].cpu())), abs=1e-9)
        # the soft path passes through the first and last cell with probability 1
        assert float(a.soft[b, 0, 0]) == pytest.approx(1.0, abs=1e-6)
        assert float(a.soft[b, -1, -1]) == pytest.approx(1.0, abs=1e-6)
        assert float(a.soft[b].min()) >= -1e-9 and float(a.soft[b].max()) <= 1 + 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_udtw_alignment(device):
    torch.manual_seed(0)
    X = torch.randn(2, 7, 3, dtype=torch.float64, device=device)
    Y = torch.randn(2, 9, 3, dtype=torch.float64, device=device)
    sx = torch.rand(2, 7, 1, dtype=torch.float64, device=device) + 0.5
    sy = torch.rand(2, 9, 1, dtype=torch.float64, device=device) + 0.5
    a = paths.udtw(X, Y, sx, sy, gamma=0.1, beta=1.0)
    d, omega = torchwarp.uDTW(gamma=0.1)(X, Y, sx, sy, beta=1.0)
    assert torch.allclose(a.distance, d) and torch.allclose(a.penalty, omega)
    assert a.soft.shape == a.variance.shape == (2, 7, 9)
    for p in a.path:
        check_monotone(p, 7, 9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("max_shift", [1, 2])
def test_jeanie_and_fvm_paths(device, max_shift):
    torch.manual_seed(0)
    q = torch.randn(2, 5, 8, 4, dtype=torch.float64, device=device)
    s = torch.randn(2, 10, 4, dtype=torch.float64, device=device)
    cost = torchwarp.euclidean_cost(q, s)

    j = paths.jeanie(q, s, gamma=0.1, max_shift=max_shift)
    assert torch.allclose(j.distance, torchwarp.jeanie_1d_from_cost(cost, 0.1, max_shift))
    assert j.soft.shape == (2, 5, 8, 10)
    for p in j.path:
        check_monotone(p, 8, 10)
        assert all(abs(b[2] - a[2]) <= max_shift for a, b in zip(p, p[1:]))

    k0 = 3
    js = paths.jeanie(q, s, gamma=0.1, max_shift=max_shift, start_view=k0)
    assert all(p[0][2] == k0 for p in js.path)
    assert torch.all(js.distance >= j.distance - 1e-9)

    f = paths.fvm(q, s, gamma=0.1)
    assert torch.allclose(f.distance, torchwarp.fvm_query_only_1d(cost, 0.1))
    for p in f.path:
        check_monotone(p, 8, 10)

    views = paths.sdtw_per_view(q, s, gamma=0.1)
    assert len(views) == 5
    for k, v in enumerate(views):
        assert all(c[2] == k for p in v.path for c in p)


def test_plot_figures():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from torchwarp import plot

    torch.manual_seed(0)
    X, Y = torch.randn(1, 12, 2), torch.randn(1, 15, 2)
    sx, sy = torch.rand(1, 12, 1) + 0.5, torch.rand(1, 15, 1) + 0.5
    fig = plot.udtw_figure(X, Y, sx, sy, gammas=(0.01, 0.1))
    assert len(fig.axes) >= 5
    plt.close(fig)

    q, s = torch.randn(5, 7, 4), torch.randn(9, 4)
    poses_q, poses_s = torch.randn(5, 7, 3, 2), torch.randn(9, 3, 2)
    fig = plot.viewpoint_figure(q, s, [-60, -30, 0, 30, 60], query_poses=poses_q,
                                support_poses=poses_s, bones=[(0, 1), (1, 2)], pose_step=2)
    assert len(fig.axes) == 3
    plt.close(fig)
    fig = plot.viewpoint_figure(q, s, [-60, -30, 0, 30, 60], max_shift=(1, 2))
    assert len(fig.axes) == 4
    plt.close(fig)
