from typing import Optional, Callable, Tuple, Any, Sequence
from functools import wraps, partial

import jax
from jax import numpy as jnp
from jax.tree_util import tree_map, tree_reduce

from .types import PyTree, Array, Key, Scalar


def center(angles: Array) -> Array:
    return ((angles + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def abs2(z: Array) -> Array:
    return jnp.real(z) ** 2 + jnp.imag(z) ** 2


def euclidean_norm(tree: PyTree) -> Scalar:
    norms2_tree = tree_map(lambda l: jnp.sum(abs2(l)), tree)
    norm2 = tree_reduce(jnp.add, norms2_tree)
    return jnp.sqrt(norm2)


def maximum_norm(tree: PyTree) -> Scalar:
    norm_tree = tree_map(lambda x: jnp.max(jnp.abs(x)), tree)
    return tree_reduce(jnp.maximum, norm_tree)


Observable = Callable[[PyTree, Array], Any]


def eval_observables(
    observables: Optional[Sequence[Observable]], params: PyTree, samples: Array
) -> Optional[Tuple[Any]]:

    if observables is None:
        return None
    else:
        return tuple(f(params, samples) for f in observables)


def eval_shape(fun, *args, has_aux=False, **kwargs):

    if has_aux:
        out, _ = jax.eval_shape(fun, *args, **kwargs)
    else:
        out = jax.eval_shape(fun, *args, **kwargs)

    return out


def maybe_split(key: Optional[Key], num: int = 2) -> Key:

    if key is not None:
        return jax.random.split(key, num)
    else:
        return jnp.zeros(shape=(num, 2), dtype=jnp.uint32)


def curry(f: Callable) -> Callable:
    return partial(partial, f)


def elementwise(f: Callable) -> Callable:

    """Apply a function elementwise to an array.

    Parameters
    ----------
    f : Callable
        The function to apply elementwise.

    Returns
    -------
    Callable
        The vectorized function.
    """

    f_vec = jax.vmap(f)

    @wraps(f)
    def wrapped(x: Array):
        x = jnp.asarray(x)
        return f_vec(x.ravel()).reshape(x.shape)  # type: ignore

    return wrapped
