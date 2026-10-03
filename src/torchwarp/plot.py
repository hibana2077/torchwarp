"""Plots in the style of the uDTW and JEANIE papers (requires matplotlib).

* ``alignment`` / ``uncertainty``: soft warping paths in white on black
  (power-normalised), and Sigma on the path with a histogram (uDTW, Fig. 2).
* ``path_3d``: paths in the (viewpoint, query time, support time) volume with
  optional skeleton glyphs (JEANIE, Fig. 7).
* ``udtw_figure`` / ``viewpoint_figure``: the full multi-panel figures.

Install the extra with ``pip install "torchwarp[vis]"``.
"""

import numpy as np
import torch

from . import paths as _paths

__all__ = ["alignment", "uncertainty", "path_3d", "udtw_figure", "viewpoint_figure",
           "PATH_COLORS", "VIEW_COLORS"]

PATH_COLORS = ["#f4a582", "#fdd49e", "#b8d8a8", "#a99de0", "#d6a8e0", "#9ecae1", "#fcbba1"]
VIEW_COLORS = ["#e41a1c", "#ff9f00", "#2ca02c", "#1f3fff", "#8b1a8b", "#17becf", "#7f7f7f"]


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError('torchwarp.plot needs matplotlib: pip install "torchwarp[vis]"') from exc
    return plt


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _image(soft, power):
    img = np.clip(_np(soft), 0, None)
    img = img / max(img.max(), 1e-12)
    return img ** power


def alignment(a, index=0, ax=None, power=0.1, title=None):
    """Soft path occupancy of a 2-D Alignment, white on black (uDTW Fig. 2a-d)."""
    plt = _plt()
    ax = ax or plt.gca()
    soft = a.soft[index]
    if soft.ndim == 3:                       # viewpoint axis: sum over views
        soft = soft.sum(0)
    ax.imshow(_image(soft, power), cmap="gray", vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    if title:
        ax.set_title(title, y=-0.16, fontsize=10)
    return ax


def uncertainty(a, index=0, ax=None, power=0.1, threshold=0.6, hist=True, title=None):
    """Sigma on the binarised path, white = high uncertainty (uDTW Fig. 2e).

    The path is the power-normalised occupancy above ``threshold``; the inset
    is the histogram of Sigma on the path.
    """
    if a.variance is None:
        raise ValueError("uncertainty() needs a uDTW Alignment (with variance)")
    plt = _plt()
    ax = ax or plt.gca()
    mask = _image(a.soft[index], power) > threshold
    var = _np(a.variance[index])
    on_path = var[mask]
    lo, hi = on_path.min(), on_path.max()
    img = np.where(mask, (var - lo) / max(hi - lo, 1e-12) * 0.85 + 0.15, 0.0)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    if hist:
        ins = ax.inset_axes([0.5, 0.62, 0.48, 0.36])
        ins.hist(on_path, bins=6, color="red", rwidth=0.8)
        ins.set_facecolor("white")
        ins.tick_params(labelsize=5, length=2, pad=1)
        ins.set_xticks([])
    if title:
        ax.set_title(title, y=-0.16, fontsize=10)
    return ax


def _glyph(ax, pose, origin, scale, color, bones, plane):
    """Draw a 2-D pose [J, 2] as a small skeleton standing at ``origin``.

    plane "tz": x-pose along the temporal axis; "kz": along the viewpoint axis.
    """
    p = _np(pose)
    p = p - p.mean(0)
    p = p / max(np.abs(p).max(), 1e-9) * scale
    k0, t0, z0 = origin
    for i, j in bones:
        xs, ys = p[[i, j], 0], p[[i, j], 1]
        if plane == "tz":
            ax.plot([k0, k0], t0 + xs, z0 + ys, color=color, lw=1.0, solid_capstyle="round")
        else:
            ax.plot(k0 + xs, [t0, t0], z0 + ys, color=color, lw=1.0, solid_capstyle="round")
    if plane == "tz":
        ax.scatter(np.full(len(p), k0), t0 + p[:, 0], z0 + p[:, 1], color=color, s=1.5)
    else:
        ax.scatter(k0 + p[:, 0], np.full(len(p), t0), z0 + p[:, 1], color=color, s=1.5)


def path_3d(alignments, labels=None, index=0, ax=None, angles=None, colors=None,
            query_poses=None, support_poses=None, bones=None, legend_loc="center left",
            title=None, elev=22, azim=-58, pose_step=1):
    """Paths in the (viewpoint, query time, support time) volume (JEANIE Fig. 7).

    Args:
        alignments: an Alignment or a list of them; each contributes the hard
            path of batch item ``index``. 2-D paths without a viewpoint are
            drawn at viewpoint 0.
        labels: legend entries, one per alignment (e.g. distances).
        angles: viewpoint labels in degrees, one per view.
        query_poses: optional [K, T, J, 2] poses drawn on the top plane, one
            per viewpoint and query time step (coloured per view).
        support_poses: optional [U, J, 2] poses drawn along the vertical axis.
        bones: list of joint index pairs for the glyphs.
        pose_step: draw a pose every ``pose_step`` time steps.
    """
    plt = _plt()
    if ax is None:
        ax = plt.figure(figsize=(5, 4)).add_subplot(projection="3d")
    if not isinstance(alignments, (list, tuple)):
        alignments = [alignments]
    colors = colors or PATH_COLORS
    first = alignments[0]
    T, U = first.soft.shape[-2:]
    top = U - 1
    angles = angles if angles is not None else first.angles
    if angles is not None:
        K = len(angles)
    else:
        K = first.soft.shape[1] if first.soft.ndim == 4 else 1

    for n, a in enumerate(alignments):
        p = np.array(a.path[index], dtype=float)
        k = p[:, 2] if p.shape[1] == 3 else np.zeros(len(p))
        lab = labels[n] if labels is not None else None
        ax.plot(k, p[:, 0], top - p[:, 1], color=colors[n % len(colors)], lw=2.2,
                alpha=0.85, label=lab, solid_capstyle="round")

    # dashed axes: viewpoints (t = 0), temporal and support time, meeting at
    # the front corner where the support poses are drawn
    kf = K - 1 + 0.6
    ax.plot([-0.4, kf], [0, 0], [top, top], "k--", lw=1.2)
    ax.plot([kf, kf], [0, T - 1], [top, top], "k--", lw=1.2)
    ax.plot([kf, kf], [0, 0], [top, 0], "k--", lw=1.2)

    if bones is not None and query_poses is not None:
        qp = _np(query_poses)
        for kk in range(qp.shape[0]):
            for tt in range(0, qp.shape[1], pose_step):
                _glyph(ax, qp[kk, tt], (kk, tt, top + 0.9 * pose_step), 0.38 * pose_step,
                       VIEW_COLORS[kk % len(VIEW_COLORS)], bones, "tz")
    if bones is not None and support_poses is not None:
        sp = _np(support_poses)
        for uu in range(0, sp.shape[0], pose_step):
            _glyph(ax, sp[uu], (kf + 0.1, -0.6 * pose_step, top - uu), 0.38 * min(pose_step, 2),
                   "black", bones, "kz")

    ax.set_xlim(-0.6, K)
    ax.set_ylim(-1, T - 0.5)
    ax.set_zlim(-0.5, top + 1.6 * pose_step)
    if angles is not None:
        ax.set_xticks(range(K))
        ax.set_xticklabels([r"${}^o$".format(int(g)) for g in angles], fontsize=8)
    else:
        ax.set_xticks([])
    ax.set_yticks(range(0, T, max(1, T // 6)))
    ax.set_yticklabels([])
    ax.set_zticks(range(0, U, max(1, U // 6)))
    ax.set_zticklabels([])
    ax.set_xlabel("Viewpoints", labelpad=4)
    ax.set_ylabel("Temporal", labelpad=-8)
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.96, 0.96, 0.96, 1.0))
        pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
    ax.view_init(elev=elev, azim=azim)
    if labels is not None:
        ax.legend(loc=legend_loc, fontsize=8, frameon=True, handlelength=2.2)
    if title:
        ax.text2D(0.5, -0.04, title, transform=ax.transAxes, ha="center", fontsize=11)
    return ax


def udtw_figure(X, Y, sigma_x, sigma_y, gammas=(0.01, 0.1), beta=1.0, index=0, power=0.1,
                threshold=0.6):
    """uDTW Fig. 2: sDTW and uDTW paths for two gammas, and uDTW uncertainty.

    X [B, N, D], Y [B, M, D], sigma_x [B, N, 1], sigma_y [B, M, 1]; batch item
    ``index`` is plotted. The sDTW panels use the same squared-Euclidean cost
    with Sigma = 1.
    """
    plt = _plt()
    fig, axes = plt.subplots(1, 2 * len(gammas) + 1, figsize=(2.1 * (2 * len(gammas) + 1), 2.5))
    ones_x, ones_y = torch.ones_like(sigma_x), torch.ones_like(sigma_y)
    col = 0
    for g in gammas:
        a = _paths.udtw(X, Y, ones_x, ones_y, gamma=g, beta=0.0)
        alignment(a, index, axes[col], power, title=r"sDTW$_{\gamma=%g}$" % g)
        col += 1
    us = []
    for g in gammas:
        us.append(_paths.udtw(X, Y, sigma_x, sigma_y, gamma=g, beta=beta))
        alignment(us[-1], index, axes[col], power, title=r"uDTW$_{\gamma=%g}$" % g)
        col += 1
    uncertainty(us[0], index, axes[col], power, threshold, title="uDTW uncert.")
    fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.98, bottom=0.14)
    return fig


def viewpoint_figure(query, support, angles, gamma=0.1, max_shift=1, metric="euclidean",
                     query_poses=None, support_poses=None, bones=None, index=0,
                     sdtw_gamma=None, fvm_gamma=None, jeanie_gamma=None, axes=None,
                     titles=None, pose_step=1):
    """JEANIE Fig. 7: soft-DTW per view, FVM and JEANIE (one path per start view).

    query [B, K, T, D] (or [K, T, D]), support [B, U, D] (or [U, D]).
    ``axes``: three existing 3-D axes to draw into (e.g. one row of a larger
    figure); ``titles``: three panel captions.
    """
    plt = _plt()
    if axes is None:
        fig = plt.figure(figsize=(15, 4.6))
        axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
        fig.subplots_adjust(wspace=0.0, left=0.0, right=1.0, top=1.02, bottom=0.07)
    fig = axes[0].figure
    titles = titles or ["(a) soft-DTW (applied per view)", "(b) FVM",
                        "(c) JEANIE ({}-max shift)".format(max_shift)]
    kw = dict(angles=angles, query_poses=query_poses, support_poses=support_poses, bones=bones,
              index=index, pose_step=pose_step)
    g_s, g_f, g_j = sdtw_gamma or gamma, fvm_gamma or gamma, jeanie_gamma or gamma

    per_view = _paths.sdtw_per_view(query, support, g_s, metric, angles)
    path_3d(per_view, ["{:.2f}".format(float(a.distance[index])) for a in per_view],
            ax=axes[0], title=titles[0], **kw)

    f = _paths.fvm(query, support, g_f, metric, angles)
    path_3d([f], [r"$d_{FVM} = %.2f$" % float(f.distance[index])], ax=axes[1],
            colors=[PATH_COLORS[3]], title=titles[1], **kw)

    starts = [_paths.jeanie(query, support, g_j, max_shift, metric, angles, start_view=k)
              for k in range(len(angles))]
    path_3d(starts, ["{:.2f}".format(float(a.distance[index])) for a in starts],
            ax=axes[2], title=titles[2], **kw)
    return fig
