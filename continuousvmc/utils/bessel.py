from functools import partial
from jax import custom_jvp  # type: ignore
from jax import numpy as jnp
from jax import lax

from .misc import elementwise
from .types import is_complex

##########################################################################################################


def _i0e_trapezoid(z):

    sign = jnp.sign(jnp.real(z))
    z_ = sign * z

    n = jnp.arange(1, 8, dtype=z.dtype)
    res = jnp.cosh(z_) + 2 * jnp.cosh(z_ * jnp.cos(n * jnp.pi / 15)).sum(axis=-1)  # type: ignore

    prefactor = jnp.exp(-jnp.real(z_))
    return prefactor * res / 15


def _i1e_trapezoid(z):

    sign = jnp.sign(jnp.real(z))
    z_ = sign * z

    n = jnp.arange(1, 8, dtype=z.dtype)
    arg = n * jnp.pi / 15  # type: ignore
    res = jnp.sinh(z_) + 2 * jnp.sum(jnp.sinh(z_ * jnp.cos(arg)) * jnp.cos(arg), axis=-1)

    prefactor = jnp.exp(-jnp.real(z_))
    return sign * prefactor * res / 15


def _i2e_trapezoid(z):

    sign = jnp.sign(jnp.real(z))
    z_ = sign * z

    n = jnp.arange(1, 8, dtype=z.dtype)
    arg = n * jnp.pi / 15  # type: ignore
    res = jnp.cosh(z_) + 2 * jnp.sum(jnp.cosh(z_ * jnp.cos(arg)) * jnp.cos(2 * arg), axis=-1)

    prefactor = jnp.exp(-jnp.real(z_))
    return prefactor * res / 15


##########################################################################################################


def _coef(n, k):
    return -(4 * n**2 - (2 * k - 1) ** 2) / (8 * k)


def _ine_asymptotic(n, z):  # n is assumed to be a nonnegative integer

    sign = jnp.sign(jnp.real(z))
    z_ = sign * z

    # fmt: off
    val = 1. + _coef(n, 1)/z * (1. + _coef(n, 2)/z * ( 1. + _coef(n, 3)/z * ( 1. + _coef(n, 4)/z * (1. + _coef(n, 5)/z * (1. + _coef(n, 6)/z * (1. + _coef(n, 7)/z * (1. + _coef(n, 8)/z)))))))
    # fmt: on

    prefactor = lax.rsqrt(2 * jnp.pi * z_)

    if is_complex(z):
        prefactor *= jnp.exp(1j * jnp.imag(z_))

    return prefactor * val * sign**n


_i0e_asymptotic = partial(_ine_asymptotic, 0)
_i1e_asymptotic = partial(_ine_asymptotic, 1)
_i2e_asymptotic = partial(_ine_asymptotic, 2)

##########################################################################################################


@elementwise
def i0e(z):
    return lax.cond(jnp.abs(z) < 20.0, _i0e_trapezoid, _i0e_asymptotic, z)


@elementwise
def i1e(z):
    return lax.cond(jnp.abs(z) < 20.0, _i1e_trapezoid, _i1e_asymptotic, z)


@elementwise
def i2e(z):
    return lax.cond(jnp.abs(z) < 20.0, _i2e_trapezoid, _i2e_asymptotic, z)


##########################################################################################################


@custom_jvp
def log_i0(z):
    return jnp.abs(z.real) + jnp.log(i0e(z))


@custom_jvp
def log_i1(z):
    return jnp.abs(z.real) + jnp.log(i1e(z))


def log_i2(z):
    return jnp.abs(z.real) + jnp.log(i2e(z))


def grad_log_i0(z):
    return jnp.exp(log_i1(z) - log_i0(z))


def hess_log_i0(z):
    term_1 = 0.5 * (1.0 + jnp.exp(log_i2(z) - log_i0(z)))
    term_2 = -jnp.exp(log_i1(z) - log_i0(z)) ** 2
    return term_1 + term_2


@log_i0.defjvp
def log_i0_jvpfun(primals, tangents):

    (z,) = primals
    (dz,) = tangents

    primal_out = log_i0(z)
    tangent_out = grad_log_i0(z) * dz

    return primal_out, tangent_out


@log_i1.defjvp
def log_i1_jvpfun(primals, tangents):

    (z,) = primals
    (dz,) = tangents

    primal_out = log_i1(z)
    tangent_out = 0.5 * (jnp.exp(log_i0(z) - primal_out) + jnp.exp(log_i2(z) - primal_out)) * dz

    return primal_out, tangent_out
