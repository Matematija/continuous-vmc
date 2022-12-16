from jax import numpy as jnp
from jax import lax
from jax import nn

from ..utils.types import Array, is_complex
from ..utils.bessel import log_i0 as _log_i0_complex, log_i1 as _log_i1_complex


def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


def _log_i0_real(x):
    return jnp.abs(x) + jnp.log(lax.bessel_i0e(x))


def log_i0(z):
    if is_complex(z):
        return _log_i0_complex(z)
    else:
        return _log_i0_real(z)


def log_i0_taylor(z):
    return z**2 / 4 - z**4 / 64 + z**6 / 576


def log_i0_sqrt_taylor(z):
    return z / 4 - z**2 / 64 + z**3 / 576


def log_i0_sqrt_pade(z):
    return (z / 4 + (7 / 576) * z**2) / (1 + z / 9)


def log_i0_pade(z):
    return (z**2 / 4 + (7 * z**4) / 576) / (1 + z**2 / 9)


def grad_log_i0_pade(z):
    return (z / 2 + z**3 / 48) / (1 + z**2 / 6)


def _log_i1_real(x):
    return jnp.abs(x) + jnp.log(lax.bessel_i1e(x))


def log_i1(z):
    if is_complex(z):
        return _log_i1_complex(z)
    else:
        return _log_i1_real(z)


def grad_log_i0(z):
    return jnp.exp(log_i1(z) - log_i0(z))


def _grad_log_i1_real(x):
    return lax.bessel_i1e(x) / lax.bessel_i0e(x)


def grad_log_i0_taylor(z):
    return z / 2 - z**3 / 16 + z**5 / 96


################################################################################


def split_real_imag(f):

    sqrt2 = jnp.sqrt(2.0)

    def split_f(z: Array, *args, **kwargs):

        if jnp.iscomplexobj(z):
            re = f(sqrt2 * z.real, *args, **kwargs)
            im = f(sqrt2 * z.imag, *args, **kwargs)
            return lax.complex(re, im) / sqrt2
        else:
            return f(z, *args, **kwargs)

    return split_f


split_relu = split_real_imag(nn.relu)
split_selu = split_real_imag(nn.selu)
split_gelu = split_real_imag(nn.gelu)
split_celu = split_real_imag(nn.celu)
split_elu = split_real_imag(nn.elu)
split_glu = split_real_imag(nn.glu)
split_leaky_relu = split_real_imag(nn.leaky_relu)
split_softplus = split_real_imag(nn.softplus)
split_swish = split_real_imag(nn.swish)
split_tanh = split_real_imag(jnp.tanh)
split_log_cosh = split_real_imag(log_cosh)
split_log_i0 = split_real_imag(_log_i0_real)
split_log_i1 = split_real_imag(_log_i1_real)
split_grad_log_i0 = split_real_imag(_grad_log_i1_real)
