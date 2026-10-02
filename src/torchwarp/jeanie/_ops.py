"""JEANIE as PyTorch custom operators plus autograd Functions.

Two layers:

1. ``torch.library.custom_op`` kernels (``torchwarp_jeanie::dp``, ``dp_backward``,
   ``features``, ``features_backward``) with fake implementations and vmap
   rules. ``torch.compile`` treats them as opaque ops, so it traces without
   graph breaks; autocast runs them in float32.
2. ``torch.autograd.Function`` wrappers (``DP``, ``Features``) in the
   ``setup_context`` style with ``generate_vmap_rule`` and ``jvp``, so every
   torch.func transform (grad, vjp, jvp, jacrev, jacfwd, hessian, vmap and
   any nesting) and ordinary autograd (create_graph=True) work.

The accumulator R is a differentiable output, like in the reference.
First-order reverse mode runs the analytic backward kernel (with dL/dR as an
extra seed). Forward mode, and differentiating *through* a backward pass
(second and higher orders), use the autograd-native implementation in
``_diff`` via ``torch.func``, so every derivative is exact. That slower path
never runs in ordinary training.

Op outputs: (out, R_diag, R[, G, q_sq, s_sq]). R_diag, G, q_sq, s_sq are
internal tensors saved for backward. R is the accumulator [B,K1,K2,T,U] when
``natural_R`` is set, otherwise an empty tensor. ``backend`` is "auto",
"cuda" or "torch"; ``metric`` is 0 (squared Euclidean) or 1 (Euclidean).
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from . import _diff, _torch_dp
from .. import _cuda
from .._layout import diag_perm, diag_perm_unt


# ------------------------------------------------------------------ helpers

def _use_kernel(t: Tensor, backend: str) -> bool:
    if backend not in ("auto", "cuda", "torch"):
        raise ValueError("backend must be 'auto', 'cuda' or 'torch'")
    if backend == "torch":
        return False
    ok = t.is_cuda and t.dtype in (torch.float32, torch.float64)
    if ok and _cuda.load("jeanie") is not None:
        return True
    if backend == "cuda":
        raise RuntimeError(
            "CUDA backend unavailable (needs CUDA float32/float64 inputs): "
            "{}".format(_cuda.build_error("jeanie"))
        )
    return False


def _to_diag(x):
    """[B,K1,K2,T,U] -> [B, T*U (diagonal-major), K1*K2]."""
    B, K1, K2, T, U = x.shape
    perm, _ = diag_perm(T, U, x.device)
    return torch.index_select(x.reshape(B, K1 * K2, T * U).transpose(1, 2), 1, perm)


def _from_diag(x, K1, K2, T, U):
    """Inverse of :func:`_to_diag`."""
    _, inv = diag_perm(T, U, x.device)
    B = x.shape[0]
    return torch.index_select(x, 1, inv).transpose(1, 2).contiguous().view(B, K1, K2, T, U)


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
        outs = outs if isinstance(outs, tuple) else (outs,)
        outs = tuple(
            o.reshape(V, o.shape[0] // V, *o.shape[1:]) if o.numel() else o.new_empty(V, 0)
            for o in outs
        )
        out_dims = tuple(0 for _ in outs)
        return (outs, out_dims) if len(outs) > 1 else (outs[0], 0)
    op.register_vmap(rule)


def _register_autocast(op):
    for device in ("cuda", "cpu"):
        torch.library.register_autocast(op, device, torch.float32)


# ------------------------------------------------- DP on a cost tensor

@torch.library.custom_op("torchwarp_jeanie::dp", mutates_args=())
def dp(cost: Tensor, s1: int, s2: int, gamma: float, backend: str,
       natural_R: bool) -> Tuple[Tensor, Tensor, Tensor]:
    B, K1, K2, T, U = cost.shape
    if _use_kernel(cost, backend):
        out, R_d = _cuda.load("jeanie").forward(_to_diag(cost), K1, K2, T, U, s1, s2, gamma)
        R = _from_diag(R_d, K1, K2, T, U) if natural_R else cost.new_empty(0)
        return out, R_d, R
    out, R = _torch_dp.jeanie_forward(cost.contiguous(), s1, s2, gamma)
    return out, _to_diag(R), (R.contiguous() if natural_R else cost.new_empty(0))


@dp.register_fake
def _(cost, s1, s2, gamma, backend, natural_R):
    B, K1, K2, T, U = cost.shape
    R = torch.empty_like(cost) if natural_R else cost.new_empty(0)
    return cost.new_empty(B), cost.new_empty(B, T * U, K1 * K2), R


@torch.library.custom_op("torchwarp_jeanie::dp_backward", mutates_args=())
def dp_backward(cost: Tensor, R_d: Tensor, grad_out: Tensor, grad_R: Optional[Tensor],
                s1: int, s2: int, gamma: float, backend: str) -> Tensor:
    B, K1, K2, T, U = cost.shape
    grad_out = grad_out.contiguous()
    if _use_kernel(cost, backend):
        gR = None if grad_R is None else _to_diag(grad_R)
        d = _cuda.load("jeanie").backward(_to_diag(cost), R_d, grad_out, gR, K1, K2, T, U, s1, s2, gamma)
        return _from_diag(d, K1, K2, T, U)
    R = _from_diag(R_d, K1, K2, T, U)
    return _torch_dp.jeanie_backward(cost, R, grad_out, s1, s2, gamma, grad_R).contiguous()


@dp_backward.register_fake
def _(cost, R_d, grad_out, grad_R, s1, s2, gamma, backend):
    return torch.empty_like(cost)


_register_vmap(dp, 1)
_register_vmap(dp_backward, 4)
_register_autocast(dp)
_register_autocast(dp_backward)


# ------------------------------------- fused DP straight from the features

@torch.library.custom_op("torchwarp_jeanie::features", mutates_args=())
def features(query: Tensor, support: Tensor, s1: int, s2: int, gamma: float, metric: int,
             backend: str, natural_R: bool
             ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    B, K1, K2, T, D = query.shape
    U, K = support.shape[1], K1 * K2
    q = query.reshape(B, K * T, D)                       # (n, t) order, no copy
    q_sq = torch.linalg.vecdot(q, q).view(B, K, T).transpose(1, 2).contiguous()  # [B,T,K]
    s_sq = torch.linalg.vecdot(support, support)
    if _use_kernel(query, backend):
        G = torch.bmm(support, q.transpose(1, 2))        # [B, U, K*T]
        perm, _ = diag_perm_unt(T, U, K, query.device)
        G = torch.index_select(G.view(B, U * K * T), 1, perm).view(B, T * U, K)
        out, R_d = _cuda.load("jeanie").fused_forward(G, q_sq, s_sq, K1, K2, T, U, s1, s2,
                                              gamma, metric)
        R = _from_diag(R_d, K1, K2, T, U) if natural_R else query.new_empty(0)
        return out, R_d, R, G, q_sq, s_sq
    G5 = torch.bmm(q, support.transpose(1, 2)).view(B, K1, K2, T, U)
    cost = _cost_from_gram(G5, q_sq, s_sq, metric)[0]
    out, R = _torch_dp.jeanie_forward(cost, s1, s2, gamma)
    R_nat = R.contiguous() if natural_R else query.new_empty(0)
    return out, _to_diag(R), R_nat, _to_diag(G5), q_sq, s_sq


def _cost_from_gram(G5, q_sq, s_sq, metric):
    """cost and raw squared distance from <q,s> [B,K1,K2,T,U] and norms."""
    B, K1, K2, T, U = G5.shape
    q2 = q_sq.transpose(1, 2).reshape(B, K1, K2, T, 1)
    sq_raw = q2 + s_sq.view(B, 1, 1, 1, U) - 2.0 * G5
    sq = sq_raw.clamp_min(0)
    return (sq.sqrt() if metric == 1 else sq), sq_raw


@features.register_fake
def _(query, support, s1, s2, gamma, metric, backend, natural_R):
    B, K1, K2, T, D = query.shape
    U, K = support.shape[1], K1 * K2
    R = query.new_empty(B, K1, K2, T, U) if natural_R else query.new_empty(0)
    return (query.new_empty(B), query.new_empty(B, T * U, K), R,
            query.new_empty(B, T * U, K), query.new_empty(B, T, K), query.new_empty(B, U))


@torch.library.custom_op("torchwarp_jeanie::features_backward", mutates_args=())
def features_backward(query: Tensor, support: Tensor, R_d: Tensor, G: Tensor, q_sq: Tensor,
                      s_sq: Tensor, grad_out: Tensor, grad_R: Optional[Tensor], s1: int,
                      s2: int, gamma: float, metric: int, backend: str
                      ) -> Tuple[Tensor, Tensor]:
    B, K1, K2, T, D = query.shape
    U, K = support.shape[1], K1 * K2
    q = query.reshape(B, K * T, D)
    grad_out = grad_out.contiguous()
    if _use_kernel(query, backend):
        gR = None if grad_R is None else _to_diag(grad_R)
        dG, dq2, ds2 = _cuda.load("jeanie").fused_backward(
            G, q_sq, s_sq, R_d, grad_out, gR, K1, K2, T, U, s1, s2, gamma, metric)
        _, inv = diag_perm_unt(T, U, K, query.device)
        dG = torch.index_select(dG.view(B, T * U * K), 1, inv).view(B, U, K * T)
        dq2 = dq2.transpose(1, 2).reshape(B, K * T, 1)  # back to (n, t) order
        dq = torch.bmm(dG.transpose(1, 2), support).addcmul_(q, dq2, value=2.0)
        ds = torch.bmm(dG, q).addcmul_(support, ds2.unsqueeze(-1), value=2.0)
        return dq.view(B, K1, K2, T, D), ds
    # portable path: same chain rule as the kernel, in PyTorch
    G5 = _from_diag(G, K1, K2, T, U)
    cost, sq_raw = _cost_from_gram(G5, q_sq, s_sq, metric)
    R = _from_diag(R_d, K1, K2, T, U)
    d_cost = _torch_dp.jeanie_backward(cost, R, grad_out, s1, s2, gamma, grad_R)
    positive = sq_raw > 0
    if metric == 1:
        dsq = torch.where(positive, 0.5 * d_cost / torch.where(positive, cost, 1.0), 0.0)
    else:
        dsq = torch.where(positive, d_cost, 0.0)
    dG = (-2.0 * dsq).reshape(B, K * T, U)
    dq = torch.bmm(dG, support).addcmul_(q, dsq.sum(-1).reshape(B, K * T, 1), value=2.0)
    ds = torch.bmm(dG.transpose(1, 2), q).addcmul_(
        support, dsq.sum((1, 2, 3)).unsqueeze(-1), value=2.0)
    return dq.view(B, K1, K2, T, D), ds


@features_backward.register_fake
def _(query, support, R_d, G, q_sq, s_sq, grad_out, grad_R, s1, s2, gamma, metric, backend):
    return torch.empty_like(query), torch.empty_like(support)


_register_vmap(features, 2)
_register_vmap(features_backward, 8)
_register_autocast(features)
_register_autocast(features_backward)


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


def _vjp_of(f, primals, cotangents):
    """J^T v for f at primals, differentiable (create_graph semantics)."""
    _, pullback = torch.func.vjp(f, *primals)
    return pullback(tuple(cotangents))


def _jvp_of(f, primals, tangents):
    """J t for f at primals. Inputs are made dense first: transforms such
    as jacrev hand over expanded (stride-0) tensors, which forward-mode
    dual tensors cannot wrap."""
    primals = tuple(p.contiguous() for p in primals)
    tangents = tuple(t.contiguous() for t in tangents)
    return torch.func.jvp(f, primals, tangents)[1]


def _r_grad(g, natural_R):
    """dL/dR when the accumulator is a real (differentiable) output."""
    return g if (natural_R and g is not None and g.numel()) else None


def _first_order_dp(s1, s2, gamma, with_R):
    """(cost, grad_out[, grad_R]) -> d_cost, as a differentiable function."""
    def f(c):
        return _diff.jeanie(c, s1, s2, gamma)

    if with_R:
        return lambda c, go, gR: _vjp_of(f, (c,), (go, gR))
    return lambda c, go: _vjp_of(lambda c: f(c)[0], (c,), (go,))


def _first_order_features(s1, s2, gamma, metric, with_R):
    """(query, support, grad_out[, grad_R]) -> (d_query, d_support)."""
    def f(q, s):
        return _diff.jeanie_features(q, s, s1, s2, gamma, metric)

    if with_R:
        return lambda q, s, go, gR: tuple(_vjp_of(f, (q, s), (go, gR)))
    return lambda q, s, go: tuple(_vjp_of(lambda q, s: f(q, s)[0], (q, s), (go,)))


class DP(torch.autograd.Function):
    """cost -> (out, R_diag, R); R is the differentiable accumulator."""

    generate_vmap_rule = True

    @staticmethod
    def forward(cost, s1, s2, gamma, backend, natural_R):
        return dp(cost, s1, s2, gamma, backend, natural_R)

    @staticmethod
    def setup_context(ctx, inputs, output):
        cost, s1, s2, gamma, backend, natural_R = inputs
        if natural_R:
            ctx.mark_non_differentiable(output[1])
        else:
            ctx.mark_non_differentiable(output[1], output[2])
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(cost, output[1])
        ctx.save_for_forward(cost)
        ctx.args = (s1, s2, gamma, backend, natural_R)

    @staticmethod
    def backward(ctx, g_out, _g_Rd, g_R):
        cost, R_d = ctx.saved_tensors
        s1, s2, gamma, backend, natural_R = ctx.args
        g_out = cost.new_zeros(cost.shape[0]) if g_out is None else g_out
        d_cost = dp_backward_apply(cost, R_d, g_out, _r_grad(g_R, natural_R),
                                  s1, s2, gamma, backend)
        return d_cost, None, None, None, None, None

    @staticmethod
    def jvp(ctx, t_cost, *_):
        (cost,) = ctx.saved_tensors
        s1, s2, gamma, _, natural_R = ctx.args
        t_out, t_R = _jvp_of(lambda c: _diff.jeanie(c, s1, s2, gamma),
                             (cost,), (_zeros(t_cost, cost),))
        return t_out, None, (t_R if natural_R else None)


class DPBackward(torch.autograd.Function):
    """First-order VJP of :class:`DP`; differentiable for higher orders."""

    generate_vmap_rule = True

    @staticmethod
    def forward(cost, R_d, g_out, g_R, s1, s2, gamma, backend):
        return dp_backward(cost, R_d, g_out, g_R, s1, s2, gamma, backend)

    @staticmethod
    def setup_context(ctx, inputs, output):
        cost, R_d, g_out, g_R, s1, s2, gamma, backend = inputs
        saved = (cost, g_out) if g_R is None else (cost, g_out, g_R)
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.args = (s1, s2, gamma)
        ctx.with_R = g_R is not None

    @staticmethod
    def backward(ctx, gg_cost):
        saved = ctx.saved_tensors
        first = _first_order_dp(*ctx.args, ctx.with_R)
        d = _vjp_of(first, saved, (_zeros(gg_cost, saved[0]),))
        d_gR = d[2] if ctx.with_R else None
        return d[0], None, d[1], d_gR, None, None, None, None

    @staticmethod
    def jvp(ctx, t_cost, _t_Rd, t_go, t_gR, *_):
        saved = ctx.saved_tensors
        tangents = (t_cost, t_go, t_gR)[:len(saved)]
        tangents = tuple(_zeros(t, p) for t, p in zip(tangents, saved))
        return _jvp_of(_first_order_dp(*ctx.args, ctx.with_R), saved, tangents)[0]


class Features(torch.autograd.Function):
    """(query, support) -> (out, R_diag, R, G, q_sq, s_sq); R differentiable."""

    generate_vmap_rule = True

    @staticmethod
    def forward(query, support, s1, s2, gamma, metric, backend, natural_R):
        return features(query, support, s1, s2, gamma, metric, backend, natural_R)

    @staticmethod
    def setup_context(ctx, inputs, output):
        query, support, s1, s2, gamma, metric, backend, natural_R = inputs
        out, R_d, R, G, q_sq, s_sq = output
        internal = (R_d, G, q_sq, s_sq) if natural_R else (R_d, R, G, q_sq, s_sq)
        ctx.mark_non_differentiable(*internal)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(query, support, R_d, G, q_sq, s_sq)
        ctx.save_for_forward(query, support)
        ctx.args = (s1, s2, gamma, metric, backend, natural_R)

    @staticmethod
    def backward(ctx, g_out, _g_Rd, g_R, _g_G, _g_q, _g_s):
        query, support, R_d, G, q_sq, s_sq = ctx.saved_tensors
        s1, s2, gamma, metric, backend, natural_R = ctx.args
        g_out = query.new_zeros(query.shape[0]) if g_out is None else g_out
        dq, ds = features_backward_apply(query, support, R_d, G, q_sq, s_sq, g_out,
                                        _r_grad(g_R, natural_R), s1, s2, gamma, metric,
                                        backend)
        return dq, ds, None, None, None, None, None, None

    @staticmethod
    def jvp(ctx, t_q, t_s, *_):
        query, support = ctx.saved_tensors
        s1, s2, gamma, metric, _, natural_R = ctx.args
        t_out, t_R = _jvp_of(
            lambda q, s: _diff.jeanie_features(q, s, s1, s2, gamma, metric),
            (query, support), (_zeros(t_q, query), _zeros(t_s, support)))
        return t_out, None, (t_R if natural_R else None), None, None, None


class FeaturesBackward(torch.autograd.Function):
    """First-order VJP of :class:`Features`; differentiable for higher orders."""

    generate_vmap_rule = True

    @staticmethod
    def forward(query, support, R_d, G, q_sq, s_sq, g_out, g_R, s1, s2, gamma, metric,
                backend):
        return features_backward(query, support, R_d, G, q_sq, s_sq, g_out, g_R,
                                 s1, s2, gamma, metric, backend)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (query, support, R_d, G, q_sq, s_sq, g_out, g_R,
         s1, s2, gamma, metric, backend) = inputs
        saved = (query, support, g_out) if g_R is None else (query, support, g_out, g_R)
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.args = (s1, s2, gamma, metric)
        ctx.with_R = g_R is not None

    @staticmethod
    def backward(ctx, gg_q, gg_s):
        saved = ctx.saved_tensors
        first = _first_order_features(*ctx.args, ctx.with_R)
        d = _vjp_of(first, saved, (_zeros(gg_q, saved[0]), _zeros(gg_s, saved[1])))
        d_gR = d[3] if ctx.with_R else None
        return (d[0], d[1], None, None, None, None, d[2], d_gR,
                None, None, None, None, None)

    @staticmethod
    def jvp(ctx, t_q, t_s, _t_Rd, _t_G, _t_qsq, _t_ssq, t_go, t_gR, *_):
        saved = ctx.saved_tensors
        tangents = (t_q, t_s, t_go, t_gR)[:len(saved)]
        tangents = tuple(_zeros(t, p) for t, p in zip(tangents, saved))
        return _jvp_of(_first_order_features(*ctx.args, ctx.with_R), saved, tangents)


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
FeaturesCompiled = _without_jvp(Features)
FeaturesBackwardCompiled = _without_jvp(FeaturesBackward)


def dp_apply(*args):
    if torch.compiler.is_compiling():
        return DPCompiled.apply(*args)
    return DP.apply(*args)


def dp_backward_apply(*args):
    if torch.compiler.is_compiling():
        return DPBackwardCompiled.apply(*args)
    return DPBackward.apply(*args)


def features_apply(*args):
    if torch.compiler.is_compiling():
        return FeaturesCompiled.apply(*args)
    return Features.apply(*args)


def features_backward_apply(*args):
    if torch.compiler.is_compiling():
        return FeaturesBackwardCompiled.apply(*args)
    return FeaturesBackward.apply(*args)
