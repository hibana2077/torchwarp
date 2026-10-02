"""uDTW as PyTorch custom operators plus autograd Functions.

Two layers:

1. ``torch.library.custom_op`` kernels (``torchwarp_udtw::dp``, ``dp_backward``,
   ``fused``, ``fused_backward``) with fake implementations and vmap rules.
   ``torch.compile`` treats them as opaque ops, so it traces without graph
   breaks; autocast runs them in float32.
2. ``torch.autograd.Function`` wrappers (``DP``, ``Fused``) in the
   ``setup_context`` style with ``generate_vmap_rule`` and ``jvp``, so every
   torch.func transform (grad, vjp, jvp, jacrev, jacfwd, hessian, vmap and
   any nesting) and ordinary autograd (create_graph=True) work.

First-order reverse mode runs the analytic backward kernel. Forward mode,
and differentiating *through* a backward pass (second and higher orders),
use the autograd-native implementation in ``_diff`` via ``torch.func``, so
every derivative is exact. That slower path never runs in ordinary training.

R, P and G are internal: diagonal-major [B, N*M] tensors saved for backward.
``backend`` is "auto", "cuda" or "torch".
"""

from typing import Tuple

import torch
from torch import Tensor

from . import _diff, _torch_dp
from .. import _cuda
from .._layout import diag_perm


# ------------------------------------------------------------------ helpers

def _use_kernel(t: Tensor, backend: str, N: int, M: int) -> bool:
    if backend not in ("auto", "cuda", "torch"):
        raise ValueError("backend must be 'auto', 'cuda' or 'torch'")
    if backend == "torch":
        return False
    ok = t.is_cuda and t.dtype in (torch.float32, torch.float64) and N <= M
    if ok and _cuda.load("udtw") is not None:
        return True
    if backend == "cuda":
        raise RuntimeError(
            "CUDA backend unavailable (needs CUDA float32/float64 inputs with N <= M): "
            "{}".format(_cuda.build_error("udtw"))
        )
    return False


def _to_diag(x):
    """[B, N, M] -> [B, N*M] diagonal-major."""
    B, N, M = x.shape
    perm, _ = diag_perm(N, M, x.device)
    return torch.index_select(x.reshape(B, N * M), 1, perm)


def _from_diag(x, N, M):
    _, inv = diag_perm(N, M, x.device)
    return torch.index_select(x, 1, inv).view(x.shape[0], N, M)


def _zeros_if_none(g, like):
    return like.new_zeros(like.shape[0]) if g is None else g.contiguous()


def _register_vmap(op, n_tensors):
    """vmap rule: fold the vmapped dimension into the leading batch dim."""
    def rule(info, in_dims, *args):
        V = info.batch_size
        folded = []
        for k, (a, d) in enumerate(zip(args, in_dims)):
            if k < n_tensors and isinstance(a, Tensor):
                a = a.unsqueeze(0).expand(V, *a.shape) if d is None else a.movedim(d, 0)
                a = a.reshape(V * a.shape[1], *a.shape[2:]).contiguous()
            folded.append(a)
        outs = op(*folded)
        outs = tuple(o.reshape(V, o.shape[0] // V, *o.shape[1:]) for o in outs)
        return outs, tuple(0 for _ in outs)
    op.register_vmap(rule)


def _register_autocast(op):
    for device in ("cuda", "cpu"):
        torch.library.register_autocast(op, device, torch.float32)


# ------------------------------------------------- DP on precomputed matrices

@torch.library.custom_op("torchwarp_udtw::dp", mutates_args=())
def dp(cost: Tensor, penalty: Tensor, gamma: float, bandwidth: float,
       backend: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    B, N, M = cost.shape
    if _use_kernel(cost, backend, N, M):
        out_r, out_p, R, P = _cuda.load("udtw").forward(
            _to_diag(cost), _to_diag(penalty), N, M, gamma, bandwidth)
        return out_r, out_p, R, P
    out_r, out_p, R, P = _torch_dp.udtw_forward(
        cost.contiguous(), penalty.contiguous(), gamma, bandwidth)
    return out_r.clone(), out_p.clone(), _to_diag(R), _to_diag(P)


@dp.register_fake
def _(cost, penalty, gamma, bandwidth, backend):
    B, N, M = cost.shape
    return (cost.new_empty(B), cost.new_empty(B),
            cost.new_empty(B, N * M), cost.new_empty(B, N * M))


@torch.library.custom_op("torchwarp_udtw::dp_backward", mutates_args=())
def dp_backward(cost: Tensor, penalty: Tensor, R: Tensor, P: Tensor,
                grad_r: Tensor, grad_p: Tensor, gamma: float, bandwidth: float,
                backend: str) -> Tuple[Tensor, Tensor]:
    B, N, M = cost.shape
    if _use_kernel(cost, backend, N, M):
        d_cost, d_pen = _cuda.load("udtw").backward(
            _to_diag(cost), _to_diag(penalty), R, P,
            grad_r.contiguous(), grad_p.contiguous(), N, M, gamma)
        return _from_diag(d_cost, N, M), _from_diag(d_pen, N, M)
    d_cost, d_pen = _torch_dp.udtw_backward(
        cost, penalty, _from_diag(R, N, M), _from_diag(P, N, M),
        grad_r.contiguous(), grad_p.contiguous(), gamma)
    return d_cost.contiguous(), d_pen.contiguous()


@dp_backward.register_fake
def _(cost, penalty, R, P, grad_r, grad_p, gamma, bandwidth, backend):
    return torch.empty_like(cost), torch.empty_like(penalty)


_register_vmap(dp, 2)
_register_vmap(dp_backward, 6)
_register_autocast(dp)
_register_autocast(dp_backward)


# ------------------------------------- fused DP straight from the features

def _features_fallback_forward(X, Y, vx, vy, gamma, beta, eps, bandwidth):
    cost, pen = _diff.matrices_from_features(X, Y, vx, vy, beta, eps)
    out_r, out_p, R, P = _torch_dp.udtw_forward(cost, pen, gamma, bandwidth)
    G = torch.bmm(X, Y.transpose(1, 2))
    x2, y2 = torch.linalg.vecdot(X, X), torch.linalg.vecdot(Y, Y)
    return out_r.clone(), out_p.clone(), _to_diag(R), _to_diag(P), _to_diag(G), x2, y2


@torch.library.custom_op("torchwarp_udtw::fused", mutates_args=())
def fused(X: Tensor, Y: Tensor, vx: Tensor, vy: Tensor, gamma: float, beta: float,
          eps: float, bandwidth: float, backend: str
          ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    N, M = X.shape[1], Y.shape[1]
    if not _use_kernel(X, backend, N, M):
        return _features_fallback_forward(X, Y, vx, vy, gamma, beta, eps, bandwidth)
    G = _to_diag(torch.bmm(X, Y.transpose(1, 2)))
    x2, y2 = torch.linalg.vecdot(X, X), torch.linalg.vecdot(Y, Y)
    out_r, out_p, R, P = _cuda.load("udtw").fused_forward(
        G, x2, y2, vx.contiguous(), vy.contiguous(), gamma, beta, eps, bandwidth)
    return out_r, out_p, R, P, G, x2, y2


@fused.register_fake
def _(X, Y, vx, vy, gamma, beta, eps, bandwidth, backend):
    B, N, M = X.shape[0], X.shape[1], Y.shape[1]
    flat = X.new_empty(B, N * M)
    return (X.new_empty(B), X.new_empty(B), flat, flat.clone(), flat.clone(),
            X.new_empty(B, N), X.new_empty(B, M))


def _features_grads(X, Y, dG, dx2, dy2):
    """Map gradients of (G = X Y^T, ||x||^2, ||y||^2) back to X and Y."""
    dX = torch.bmm(dG, Y).addcmul_(X, dx2.unsqueeze(-1), value=2.0)
    dY = torch.bmm(dG.transpose(1, 2), X).addcmul_(Y, dy2.unsqueeze(-1), value=2.0)
    return dX, dY


@torch.library.custom_op("torchwarp_udtw::fused_backward", mutates_args=())
def fused_backward(X: Tensor, Y: Tensor, vx: Tensor, vy: Tensor, R: Tensor, P: Tensor,
                   G: Tensor, x2: Tensor, y2: Tensor, grad_r: Tensor, grad_p: Tensor,
                   gamma: float, beta: float,
                   eps: float, bandwidth: float, backend: str
                   ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    N, M = X.shape[1], Y.shape[1]
    grad_r, grad_p = grad_r.contiguous(), grad_p.contiguous()
    if _use_kernel(X, backend, N, M):
        dG, dx2, dy2, dvx, dvy = _cuda.load("udtw").fused_backward(
            G, x2, y2, vx.contiguous(), vy.contiguous(), R, P, grad_r, grad_p,
            gamma, beta, eps)
        dX, dY = _features_grads(X, Y, _from_diag(dG, N, M), dx2, dy2)
        return dX, dY, dvx, dvy
    # portable path: same chain rule as the kernel, in PyTorch
    Gm = _from_diag(G, N, M)
    sq_raw = x2.unsqueeze(2) + y2.unsqueeze(1) - 2.0 * Gm
    var_raw = 0.5 * (vx.unsqueeze(2) + vy.unsqueeze(1))
    sq, var = sq_raw.clamp_min(0), var_raw.clamp_min(eps)
    cost, pen = sq / var, beta * torch.log(var)
    d_cost, d_pen = _torch_dp.udtw_backward(
        cost, pen, _from_diag(R, N, M), _from_diag(P, N, M), grad_r, grad_p, gamma)
    dsq = torch.where(sq_raw > 0, d_cost / var, 0.0)
    dvar = torch.where(var_raw > eps, (beta * d_pen - d_cost * cost) / var, 0.0)
    dX, dY = _features_grads(X, Y, -2.0 * dsq, dsq.sum(2), dsq.sum(1))
    return dX, dY, 0.5 * dvar.sum(2), 0.5 * dvar.sum(1)


@fused_backward.register_fake
def _(X, Y, vx, vy, R, P, G, x2, y2, grad_r, grad_p, gamma, beta, eps, bandwidth, backend):
    return (torch.empty_like(X), torch.empty_like(Y),
            torch.empty_like(vx), torch.empty_like(vy))


_register_vmap(fused, 4)
_register_vmap(fused_backward, 11)
_register_autocast(fused)
_register_autocast(fused_backward)


# ------------------------------------------------------- autograd Functions
#
# Derivative routing:
#   reverse, first order  -> *_backward custom op (analytic kernel)
#   forward mode (jvp)    -> torch.func.jvp of the _diff implementation
#   through a backward    -> torch.func.vjp/jvp of the _diff implementation
# torch.func.vjp/jvp compose with ordinary autograd and with every
# torch.func transform, so any nesting stays exact.


def _zeros(g, like):
    return torch.zeros_like(like) if g is None else g


def _jvp_of(f, primals, tangents):
    """J t for f at primals. Inputs are made dense first: transforms such
    as jacrev hand over expanded (stride-0) tensors, which forward-mode
    dual tensors cannot wrap."""
    primals = tuple(p.contiguous() for p in primals)
    tangents = tuple(t.contiguous() for t in tangents)
    return torch.func.jvp(f, primals, tangents)[1]


def _vjp_of(f, primals, cotangents):
    """J^T v for f at primals, differentiable (create_graph semantics)."""
    _, pullback = torch.func.vjp(f, *primals)
    return pullback(tuple(cotangents))


def _first_order(f, n):
    """(x_1..x_n, v_1..v_k) -> J_f(x)^T v, itself differentiable."""
    def g(*args):
        return _vjp_of(f, args[:n], args[n:])
    return g


class DP(torch.autograd.Function):
    """(cost, penalty) -> (distance, penalty_sum, R, P); R and P internal."""

    generate_vmap_rule = True

    @staticmethod
    def forward(cost, penalty, gamma, bandwidth, backend):
        return dp(cost, penalty, gamma, bandwidth, backend)

    @staticmethod
    def setup_context(ctx, inputs, output):
        cost, penalty, gamma, bandwidth, backend = inputs
        ctx.mark_non_differentiable(output[2], output[3])
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(cost, penalty, output[2], output[3])
        ctx.save_for_forward(cost, penalty)
        ctx.args = (gamma, bandwidth, backend)

    @staticmethod
    def backward(ctx, g_r, g_p, _g_R, _g_P):
        cost, penalty, R, P = ctx.saved_tensors
        gamma, bandwidth, backend = ctx.args
        B = cost.shape[0]
        g_r = cost.new_zeros(B) if g_r is None else g_r
        g_p = cost.new_zeros(B) if g_p is None else g_p
        d_cost, d_pen = dp_backward_apply(cost, penalty, R, P, g_r, g_p,
                                         gamma, bandwidth, backend)
        return d_cost, d_pen, None, None, None

    @staticmethod
    def jvp(ctx, t_cost, t_pen, *_):
        cost, penalty = ctx.saved_tensors
        gamma, bandwidth, _ = ctx.args
        f = lambda c, q: _diff.udtw(c, q, gamma, bandwidth)
        t_r, t_p = _jvp_of(f, (cost, penalty), (_zeros(t_cost, cost), _zeros(t_pen, penalty)))
        return t_r, t_p, None, None


class DPBackward(torch.autograd.Function):
    """First-order VJP of :class:`DP`; differentiable for higher orders."""

    generate_vmap_rule = True

    @staticmethod
    def forward(cost, penalty, R, P, g_r, g_p, gamma, bandwidth, backend):
        return dp_backward(cost, penalty, R, P, g_r, g_p, gamma, bandwidth, backend)

    @staticmethod
    def setup_context(ctx, inputs, output):
        cost, penalty, R, P, g_r, g_p, gamma, bandwidth, backend = inputs
        ctx.save_for_backward(cost, penalty, g_r, g_p)
        ctx.save_for_forward(cost, penalty, g_r, g_p)
        ctx.args = (gamma, bandwidth)

    @staticmethod
    def _first(ctx):
        gamma, bandwidth = ctx.args
        return _first_order(lambda c, q: _diff.udtw(c, q, gamma, bandwidth), 2)

    @staticmethod
    def backward(ctx, gg_cost, gg_pen):
        cost, penalty, g_r, g_p = ctx.saved_tensors
        d = _vjp_of(DPBackward._first(ctx), (cost, penalty, g_r, g_p),
                    (_zeros(gg_cost, cost), _zeros(gg_pen, penalty)))
        return d[0], d[1], None, None, d[2], d[3], None, None, None

    @staticmethod
    def jvp(ctx, t_cost, t_pen, _t_R, _t_P, t_gr, t_gp, *_):
        cost, penalty, g_r, g_p = ctx.saved_tensors
        primals = (cost, penalty, g_r, g_p)
        tangents = tuple(_zeros(t, p) for t, p in zip((t_cost, t_pen, t_gr, t_gp), primals))
        return tuple(_jvp_of(DPBackward._first(ctx), primals, tangents))


class Fused(torch.autograd.Function):
    """(X, Y, vx, vy) -> (distance, penalty_sum, R, P, G); R, P, G internal."""

    generate_vmap_rule = True

    @staticmethod
    def forward(X, Y, vx, vy, gamma, beta, eps, bandwidth, backend):
        return fused(X, Y, vx, vy, gamma, beta, eps, bandwidth, backend)

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, Y, vx, vy, gamma, beta, eps, bandwidth, backend = inputs
        ctx.mark_non_differentiable(*output[2:])
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(X, Y, vx, vy, *output[2:])
        ctx.save_for_forward(X, Y, vx, vy)
        ctx.args = (gamma, beta, eps, bandwidth, backend)

    @staticmethod
    def _f(ctx):
        gamma, beta, eps, bandwidth, _ = ctx.args
        return lambda x, y, a, b: _diff.udtw_features(x, y, a, b, gamma, beta, eps, bandwidth)

    @staticmethod
    def backward(ctx, g_r, g_p, *_internal):
        X, Y, vx, vy, R, P, G, x2, y2 = ctx.saved_tensors
        gamma, beta, eps, bandwidth, backend = ctx.args
        B = X.shape[0]
        g_r = X.new_zeros(B) if g_r is None else g_r
        g_p = X.new_zeros(B) if g_p is None else g_p
        grads = fused_backward_apply(X, Y, vx, vy, R, P, G, x2, y2, g_r, g_p,
                                    gamma, beta, eps, bandwidth, backend)
        return (*grads, None, None, None, None, None)

    @staticmethod
    def jvp(ctx, t_X, t_Y, t_vx, t_vy, *_):
        primals = ctx.saved_tensors
        tangents = tuple(_zeros(t, p) for t, p in zip((t_X, t_Y, t_vx, t_vy), primals))
        t_r, t_p = _jvp_of(Fused._f(ctx), primals, tangents)
        return t_r, t_p, None, None, None, None, None


class FusedBackward(torch.autograd.Function):
    """First-order VJP of :class:`Fused`; differentiable for higher orders."""

    generate_vmap_rule = True

    @staticmethod
    def forward(X, Y, vx, vy, R, P, G, x2, y2, g_r, g_p, gamma, beta, eps, bandwidth,
                backend):
        return fused_backward(X, Y, vx, vy, R, P, G, x2, y2, g_r, g_p,
                              gamma, beta, eps, bandwidth, backend)

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, Y, vx, vy, R, P, G, x2, y2, g_r, g_p, gamma, beta, eps, bandwidth, backend = inputs
        ctx.save_for_backward(X, Y, vx, vy, g_r, g_p)
        ctx.save_for_forward(X, Y, vx, vy, g_r, g_p)
        ctx.args = (gamma, beta, eps, bandwidth)

    @staticmethod
    def _first(ctx):
        gamma, beta, eps, bandwidth = ctx.args
        return _first_order(
            lambda x, y, a, b: _diff.udtw_features(x, y, a, b, gamma, beta, eps, bandwidth), 4)

    @staticmethod
    def backward(ctx, gg_X, gg_Y, gg_vx, gg_vy):
        X, Y, vx, vy, g_r, g_p = ctx.saved_tensors
        cot = tuple(_zeros(g, p) for g, p in zip((gg_X, gg_Y, gg_vx, gg_vy), (X, Y, vx, vy)))
        d = _vjp_of(FusedBackward._first(ctx), (X, Y, vx, vy, g_r, g_p), cot)
        return (d[0], d[1], d[2], d[3], None, None, None, None, None, d[4], d[5],
                None, None, None, None, None)

    @staticmethod
    def jvp(ctx, t_X, t_Y, t_vx, t_vy, _t_R, _t_P, _t_G, _t_x2, _t_y2, t_gr, t_gp, *_):
        primals = ctx.saved_tensors
        tangents = tuple(_zeros(t, p) for t, p in
                         zip((t_X, t_Y, t_vx, t_vy, t_gr, t_gp), primals))
        return tuple(_jvp_of(FusedBackward._first(ctx), primals, tangents))


# ------------------------------------------------- torch.compile variants
#
# Dynamo cannot trace an autograd.Function that defines ``jvp``. Each
# Function above therefore gets a twin without ``jvp``; ``apply`` picks the
# twin while torch.compile is tracing and the full class otherwise (eager
# mode and torch.func, where forward mode needs ``jvp``).

def _without_jvp(cls):
    ns = {k: v for k, v in vars(cls).items()
          if k not in ("jvp", "__dict__", "__weakref__")}
    return type(cls.__name__ + "Compiled", (torch.autograd.Function,), ns)


DPCompiled = _without_jvp(DP)
DPBackwardCompiled = _without_jvp(DPBackward)
FusedCompiled = _without_jvp(Fused)
FusedBackwardCompiled = _without_jvp(FusedBackward)


def dp_apply(*args):
    if torch.compiler.is_compiling():
        return DPCompiled.apply(*args)
    return DP.apply(*args)


def dp_backward_apply(*args):
    if torch.compiler.is_compiling():
        return DPBackwardCompiled.apply(*args)
    return DPBackward.apply(*args)


def fused_apply(*args):
    if torch.compiler.is_compiling():
        return FusedCompiled.apply(*args)
    return Fused.apply(*args)


def fused_backward_apply(*args):
    if torch.compiler.is_compiling():
        return FusedBackwardCompiled.apply(*args)
    return FusedBackward.apply(*args)
