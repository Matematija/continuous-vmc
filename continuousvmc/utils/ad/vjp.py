from typing import Callable, Tuple, Any, Union

import jax
from jax.tree_util import tree_map

from jax import numpy as jnp

from ..misc import eval_shape
from ..types import is_complex, tree_is_complex

## The following code is adapted from:
## https://github.com/netket/netket/tree/master/netket/jax


def _vjp_cc(
    fun: Callable, *primals, has_aux: bool = False
) -> Union[Tuple[Any, Callable], Tuple[Any, Callable, Any]]:

    if has_aux:
        out, _vjp_fun, aux = jax.vjp(fun, *primals, has_aux=True)
    else:
        out, _vjp_fun = jax.vjp(fun, *primals, has_aux=False)

    def vjp_fun(ȳ):
        ȳ = jnp.asarray(ȳ, dtype=out.dtype)
        dȳ = _vjp_fun(ȳ)
        return dȳ

    if has_aux:
        return out, vjp_fun, aux
    else:
        return out, vjp_fun


def _vjp_rr(
    fun: Callable, *primals, has_aux: bool = False
) -> Union[Tuple[Any, Callable], Tuple[Any, Callable, Any]]:

    if has_aux:
        primals_out, _vjp_fun, aux = jax.vjp(fun, *primals, has_aux=True)
    else:
        primals_out, _vjp_fun = jax.vjp(fun, *primals, has_aux=False)

    def vjp_fun(ȳ):

        """
        function computing the vjp product for a R->R function.
        """

        if not is_complex(ȳ):
            out = _vjp_fun(jnp.asarray(ȳ, dtype=primals_out.dtype))

        else:

            out_r = _vjp_fun(jnp.asarray(ȳ.real, dtype=primals_out.dtype))
            out_i = _vjp_fun(jnp.asarray(ȳ.imag, dtype=primals_out.dtype))

            out = tree_map(lambda re, im: re + 1j * im, out_r, out_i)

        return out

    if has_aux:
        return primals_out, vjp_fun, aux
    else:
        return primals_out, vjp_fun


def vjp_rc(
    fun: Callable, *primals, has_aux: bool = False
) -> Union[Tuple[Any, Callable], Tuple[Any, Callable, Any]]:

    if has_aux:

        def real_fun(*primals):
            val, aux = fun(*primals)
            return val.real, aux

        def imag_fun(*primals):
            val, aux = fun(*primals)
            return val.imag, aux

        vals_r, vjp_r_fun, aux = jax.vjp(real_fun, *primals, has_aux=True)
        vals_j, vjp_j_fun, _ = jax.vjp(imag_fun, *primals, has_aux=True)

    else:
        real_fun = lambda *primals: fun(*primals).real
        imag_fun = lambda *primals: fun(*primals).imag

        vals_r, vjp_r_fun = jax.vjp(real_fun, *primals, has_aux=False)
        vals_j, vjp_j_fun = jax.vjp(imag_fun, *primals, has_aux=False)

    primals_out = vals_r + 1j * vals_j

    def vjp_fun(ȳ):

        """
        function computing the vjp product for a R->C function.
        """

        ȳ_r = ȳ.real
        ȳ_j = ȳ.imag

        vr_jr = vjp_r_fun(jnp.asarray(ȳ_r, dtype=vals_r.dtype))
        vj_jr = vjp_r_fun(jnp.asarray(ȳ_j, dtype=vals_r.dtype))
        vr_jj = vjp_j_fun(jnp.asarray(ȳ_r, dtype=vals_j.dtype))
        vj_jj = vjp_j_fun(jnp.asarray(ȳ_j, dtype=vals_j.dtype))

        r = tree_map(lambda re, im: re + 1j * im, vr_jr, vj_jr)
        i = tree_map(lambda re, im: re + 1j * im, vr_jj, vj_jj)
        out = tree_map(lambda re, im: re + 1j * im, r, i)

        return out

    if has_aux:
        return primals_out, vjp_fun, aux
    else:
        return primals_out, vjp_fun


def vjp(
    fun: Callable, *primals, has_aux: bool = False
) -> Union[Tuple[Any, Callable], Tuple[Any, Callable, Any]]:

    out_shape = eval_shape(fun, *primals, has_aux=has_aux)

    if tree_is_complex(primals):

        if is_complex(out_shape):  # C -> C
            return _vjp_cc(fun, *primals, has_aux=has_aux)
        else:  # C -> R
            return _vjp_cc(fun, *primals, has_aux=has_aux)

    else:

        if is_complex(out_shape):  # R -> C
            return vjp_rc(fun, *primals, has_aux=has_aux)
        else:  # R -> R
            return _vjp_rr(fun, *primals, has_aux=has_aux)
