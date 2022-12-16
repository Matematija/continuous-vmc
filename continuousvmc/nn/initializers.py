from typing import Callable, Sequence, Any
from functools import wraps

from jax import numpy as jnp
from jax import lax

from jax.nn import initializers

from ..utils.types import Key, default_real, is_complex_dtype


def _wrap_init(initializer: Callable):
    @wraps(initializer)
    def wrapped(*args, zero_phase: bool = True, **kwargs):

        init_fn = initializer(*args, **kwargs)

        def init(key: Key, shape: Sequence[int], dtype: Any = None):

            if dtype is None:
                dtype = default_real()

            x = init_fn(key, shape, dtype)

            if zero_phase and is_complex_dtype(dtype):
                x_re = jnp.real(x)
                x = lax.complex(x_re, jnp.zeros_like(x_re))

            return x

        return init

    return wrapped


variance_scaling = _wrap_init(initializers.variance_scaling)
lecun_normal = _wrap_init(initializers.lecun_normal)
lecun_uniform = _wrap_init(initializers.lecun_uniform)
normal = _wrap_init(initializers.normal)
uniform = _wrap_init(initializers.uniform)
orthogonal = _wrap_init(initializers.orthogonal)
zeros = initializers.zeros
ones = initializers.ones


################################################################################


def default_kernel_init(
    scale=0.1,
    distribution="truncated_normal",
    mode="fan_in",
    in_axis=-1,
    out_axis=-2,
    zero_phase=True,
) -> Callable:

    return variance_scaling(
        scale=scale,
        distribution=distribution,
        mode=mode,
        in_axis=in_axis,
        out_axis=out_axis,
        zero_phase=zero_phase,
    )
