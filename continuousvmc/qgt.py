from typing import Any, Callable, Tuple, Optional, Union
from functools import partial
from warnings import warn

import jax
from jax import lax
from jax import numpy as jnp
from jax import scipy as jsp
from jax.tree_util import Partial, tree_map
from jax._src.numpy.util import _promote_dtypes_inexact

from flax import struct

from .utils.ad import grad
from .utils.tree import basis_of, tree_destructure, tree_conj, xpay
from .utils.chunk import vmap_chunked
from .utils.misc import abs2
from .utils.types import PyTree, Array, Ansatz, Scalar, is_real, real_dtype

#############################################################################


@struct.dataclass
class QGT:

    batched_grad: Callable = struct.field(repr=False)
    solve_fn: Callable = struct.field(repr=False)
    scale: bool = struct.field(pytree_node=False, default=False)
    eps: Optional[Scalar] = None

    def to_dense(self, params: PyTree, samples: Array):
        return _qgt_to_dense(self, params, samples, scale=self.scale)

    def solve(
        self, params: PyTree, samples: Array, rhs: PyTree, *args, **kwargs
    ) -> Tuple[PyTree, Any]:

        kwargs.pop("x0", None)

        return solve_dense(self, params, samples, rhs, *args, **kwargs)


#############################################################################


def get_scale(oc: Array) -> Array:
    scale2 = jnp.mean(abs2(oc), axis=0)
    # scale2, _ = mpi_allreduce_mean(scale2)
    return jnp.sqrt(scale2)


def centered_jacobian(
    batched_grad: Callable, params: PyTree, samples: Array, scale: bool = False
) -> Array:

    o_tree = batched_grad(params, samples)
    os, _ = tree_destructure(o_tree, keep_batch_dim=True)

    os_mean = os.mean(axis=0, keepdims=True)
    # os_mean, _ = mpi_allreduce_mean(os_mean)
    oc = os - os_mean

    scale_fac = lax.cond(
        scale, get_scale, lambda _: jnp.ones(oc.shape[-1], dtype=real_dtype(oc.dtype)), oc
    )

    oc /= scale_fac[None, :]
    oc /= jnp.sqrt(samples.shape[0])

    return oc, scale_fac


def full_matrix(oc: Array, eps: Optional[Scalar] = None) -> Array:

    S_mat = oc.T.conj() @ oc

    if eps is not None:
        S_mat = S_mat.at[jnp.diag_indices_from(S_mat)].add(eps)

    return S_mat


@jax.jit
def _qgt_to_dense(S: QGT, params: PyTree, samples: Array) -> Array:
    oc, *_ = centered_jacobian(S.batched_grad, params, samples, scale=False)
    S_mat = full_matrix(oc, eps=None)
    return S_mat


#############################################################################


@jax.jit
def solve_dense(S: QGT, params: PyTree, samples: Array, rhs: PyTree, *args):

    oc, scale_fac = centered_jacobian(S.batched_grad, params, samples, scale=S.scale)
    b, re = tree_destructure(rhs)

    sol, info = S.solve_fn(oc, b, *args, eps=S.eps)
    sol = lax.cond(S.scale, lambda x: x / scale_fac, lambda x: x, sol)

    return re(sol), info


def _solve_shift(oc: Array, rhs: Array, *_, eps: Optional[Scalar] = None) -> Tuple[PyTree, Any]:

    S_mat = full_matrix(oc, eps=eps)

    if is_real(rhs):
        S_mat = jnp.real(S_mat)

    sol = jsp.linalg.solve(S_mat, rhs, assume_a="pos")
    return sol, None


#############################################################################


def _default_inv_fn(s, acond, rcond):

    cutoff = jnp.finfo(s.dtype).eps

    if acond is not None:
        acond = jnp.maximum(cutoff, acond)

    cutoff = jnp.maximum(acond, rcond * s[0])

    mask = s >= cutoff
    safe_s = jnp.where(mask, s, 1)
    s_inv = jnp.where(mask, 1 / safe_s, 0)

    rank = mask.sum()
    return s_inv, rank


def lstsq(
    a: Array,
    b: Array,
    rcond: Scalar = 0.0,
    acond: Optional[Scalar] = None,
    *,
    hermitian: bool = False,
    inv_fn: Optional[Callable] = None,
):

    a, b = _promote_dtypes_inexact(a, b)

    if a.shape[0] != b.shape[0]:
        raise ValueError("Leading dimensions of input arrays must match")

    b_orig_ndim = b.ndim

    if b_orig_ndim == 1:
        b = b[:, None]

    if a.ndim != 2:
        raise TypeError(f"{a.ndim}-dimensional array given. Array must be two-dimensional")

    if b.ndim != 2:
        raise TypeError(f"{b.ndim}-dimensional array given. Array must be one or two-dimensional")

    u, s, vt = jnp.linalg.svd(a, full_matrices=False, hermitian=hermitian)

    if inv_fn is None:
        inv_fn = partial(_default_inv_fn, acond=acond, rcond=rcond)

    s_inv, *inv_info = inv_fn(s)
    s_inv = s_inv[:, None]

    precision = lax.Precision.HIGHEST
    uTb = jnp.matmul(u.T.conj(), b, precision=precision)
    x = jnp.matmul(vt.T.conj(), s_inv * uTb, precision=precision)

    b_estimate = jnp.matmul(a, x, precision=precision)
    resid = jnp.linalg.norm(b - b_estimate, axis=0) ** 2

    if b_orig_ndim == 1:
        x = x.ravel()

    return x, *inv_info, resid, s


def _solve_svd(
    oc: Array,
    rhs: Array,
    *_,
    eps: Optional[Scalar] = None,
    rcond: Scalar = 0.0,
    acond: Optional[Scalar] = None,
    inv_fn: Optional[Callable] = None,
) -> Tuple[PyTree, Any]:

    S_mat = full_matrix(oc, eps=eps)

    if is_real(rhs):
        S_mat = jnp.real(S_mat)

    sol, *info = lstsq(S_mat, rhs, hermitian=True, inv_fn=inv_fn, rcond=rcond, acond=acond)

    return sol, tuple(info)


#############################################################################


def _snr(o_tilde, b_tilde, Ec, eps=None):

    if eps is None:
        eps = jnp.finfo(real_dtype(Ec.dtype)).eps

    numerator = jnp.abs(b_tilde) * jnp.sqrt(o_tilde.shape[0])

    denom_1 = jnp.sum(abs2(o_tilde.conj() * Ec[:, None]), axis=0)
    denom_2 = abs2(b_tilde)
    denominator = denom_1 - denom_2

    denominator = jnp.maximum(denominator, eps)
    denominator = jnp.sqrt(denominator)

    return numerator / denominator


def _default_reg_fn(x, rcond, acond, exponent):

    cutoff = jnp.finfo(real_dtype(x.dtype)).eps

    if acond is not None:
        cutoff = jnp.maximum(cutoff, acond)

    cutoff = jnp.maximum(cutoff, rcond * jnp.max(x))

    return 1 / (1 + (cutoff / x) ** exponent)


def _solve_snr(
    oc: Array,
    rhs: Array,
    Ec: Array,
    *_,
    eps: Optional[Scalar] = None,
    snr_cutoff: Scalar = 4.0,
    exponent: Scalar = 6.0,
    svd_rcond: Scalar = 0.0,
    svd_acond: Optional[Scalar] = None,
    reg_fn: Optional[Callable] = None,
) -> Tuple[PyTree, Any]:

    S_mat = full_matrix(oc, eps=eps)

    if is_real(rhs):
        S_mat = jnp.real(S_mat)

    s2, V = jnp.linalg.eigh(S_mat)
    Vd = V.T.conj()

    b_tilde = Vd @ rhs
    o_tilde = oc @ Vd

    snr = _snr(o_tilde, b_tilde, Ec)

    if reg_fn is None:
        reg_fn = partial(_default_reg_fn, exponent=exponent)

    svd_reg = reg_fn(s2, rcond=svd_rcond, acond=svd_acond)
    snr_reg = reg_fn(snr, rcond=0.0, acond=snr_cutoff)

    cutoff = 10 * jnp.finfo(s2.dtype).eps
    s2_safe = jnp.maximum(s2, cutoff)

    sol = V @ (svd_reg * snr_reg / s2_safe)
    effective_rank = jnp.sum(svd_reg * snr_reg)

    info = (effective_rank, svd_reg, snr_reg, snr)

    return sol, info


#############################################################################


@struct.dataclass
class IterativeQGT:

    batched_apply: Callable = struct.field(repr=False)
    solver: Callable = struct.field(repr=False)
    eps: Scalar = 0.0

    def to_dense(self, params: PyTree, samples: Array, chunk_size: int = None) -> Array:
        return _iterative_qgt_to_dense(self, params, samples, chunk_size=chunk_size)

    def solve(self, params: PyTree, samples: Array, rhs: PyTree, **kwargs) -> Tuple[PyTree, Any]:
        return self.solver(self, params, samples, rhs, **kwargs)


@partial(jax.jit, static_argnames="chunk_size")
def _iterative_qgt_to_dense(
    S: IterativeQGT, params: PyTree, samples: Array, chunk_size: Optional[int] = None
) -> Array:

    svp = make_svp_fn(Partial(S.batched_apply), params, samples, S.eps)
    basis = basis_of(params)

    S_tree = vmap_chunked(svp, chunk_size=chunk_size)(basis)
    S_mat, _ = tree_destructure(S_tree, keep_batch_dim=True)

    return S_mat.at[jnp.diag_indices_from(S_mat)].add(-S.eps).conj()


#############################################################################


@jax.jit
def _svp(fwd: Partial, eps: Scalar, v: Array):

    back = jax.linear_transpose(fwd, v)

    w = fwd(v)
    w /= w.size

    w_mean = w.mean(axis=0, keepdims=True)
    wc = w - w_mean

    (res,) = tree_conj(back(wc.conj()))

    if is_real(v):
        res = tree_map(res, lambda l: 2 * jnp.real(l))

    return xpay(res, v, eps)


@jax.jit
def make_svp_fn(batched_apply: Partial, params: PyTree, samples: Array, eps: Scalar):
    _, fwd = jax.linearize(lambda p: batched_apply(p, samples), params)
    return Partial(_svp, fwd, eps)


#############################################################################


def _wrap_iterative_jax_solver(solver, **kwargs):

    solver = partial(solver, **kwargs)

    @jax.jit
    def wrapped_solver(
        S: IterativeQGT, params: PyTree, samples: Array, rhs: PyTree, x0: Optional[PyTree] = None
    ):

        svp = make_svp_fn(S.batched_apply, params, samples, S.eps)

        if x0 is None:
            x0 = tree_map(jnp.zeros_like, rhs)

        sol, info = solver(svp, rhs, x0=x0)

        return sol, info

    return wrapped_solver


#############################################################################

_DENSE_SLOVERS = {"shift": _solve_shift, "svd": _solve_svd, "snr": _solve_snr}

_SPARSE_SOLVERS = {
    "cg": jsp.sparse.linalg.cg,
    "gmres": jsp.sparse.linalg.gmres,
    "bicgstab": jsp.sparse.linalg.bicgstab,
}

#############################################################################


def is_dense_solver(name: str) -> bool:

    name = name.lower()

    if name in _DENSE_SLOVERS:
        return True
    elif name in _SPARSE_SOLVERS:
        return False
    else:
        raise ValueError(f"Unknown solver: {name}")


def is_sparse_solver(name: str) -> bool:
    return not is_dense_solver(name)


def QuantumGeometricTensor(
    logpsi: Union[Ansatz, Callable],
    solver: str = "shift",
    eps: Optional[Scalar] = None,
    scale: bool = False,
    chunk_size: Optional[int] = None,
    **solver_kwargs,
) -> Union[QGT, IterativeQGT]:

    apply_fn = logpsi.apply if hasattr(logpsi, "apply") else logpsi

    solver = solver.lower()

    solver_kwargs = {
        key: Partial(val) if callable(val) else val for key, val in solver_kwargs.items()
    }

    if solver.lower() in _DENSE_SLOVERS:

        batched_grad = vmap_chunked(
            grad(apply_fn, argnums=0), in_axes=(None, 0), chunk_size=chunk_size
        )

        solve_fn = Partial(_DENSE_SLOVERS[solver], **solver_kwargs)
        return QGT(Partial(batched_grad), solve_fn, scale=scale, eps=eps)

    elif solver.lower() in _SPARSE_SOLVERS:

        batched_apply = vmap_chunked(apply_fn, in_axes=(None, 0), chunk_size=chunk_size)
        eps = eps if eps is not None else 0.0

        solver_fn = _wrap_iterative_jax_solver(_SPARSE_SOLVERS[solver], **solver_kwargs)
        return IterativeQGT(Partial(batched_apply), Partial(solver_fn), eps=eps)
