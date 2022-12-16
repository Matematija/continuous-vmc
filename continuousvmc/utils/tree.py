from typing import Any, Tuple, Callable
from functools import partial, wraps

import numpy as np

import jax
from jax import numpy as jnp
from jax.tree_util import (
    tree_reduce,
    tree_map,
    tree_leaves,
    tree_unflatten,
    tree_flatten,
)

from .types import PyTree, Array, Key


def tree_dot(a: PyTree, b: PyTree) -> PyTree:
    return tree_reduce(jnp.add, tree_map(jnp.sum, tree_map(jnp.multiply, a, b)))


def xpay(x: PyTree, y: PyTree, a: Any) -> PyTree:
    return tree_map(lambda x_, y_: x_ + a * y_, x, y)


def tree_destructure(
    tree: PyTree, keep_batch_dim: bool = False
) -> Tuple[Array, Callable[[Array], PyTree]]:

    leaves, treedef = tree_flatten(tree)
    flat, rebuild = jax.vjp(_ravel_list if not keep_batch_dim else _ravel_batched_list, *leaves)
    restructure = lambda flat: tree_unflatten(treedef, rebuild(flat))

    return flat, restructure  # type: ignore


def _ravel_list(*l):
    return jnp.concatenate([jnp.ravel(leaf) for leaf in l], axis=0) if l else jnp.array([])


def _ravel_batched_list(*l):
    return (
        jnp.concatenate([leaf.reshape(leaf.shape[0], -1) for leaf in l], axis=1)
        if l
        else jnp.array([])
    )


def basis_of(tree: PyTree):
    flat, re = tree_destructure(tree)
    return jax.vmap(re)(jnp.eye(flat.size, dtype=flat.dtype))


def tree_size(tree: PyTree) -> int:
    return sum(l.size for l in tree_leaves(tree))


def tree_rebuild(tree: PyTree) -> Callable:

    leaves, treedef = tree_flatten(tree)

    shapes = [l.shape for l in leaves]
    sizes = [l.size for l in leaves]
    indices = np.cumsum(sizes)[:-1]

    def rebuild(arr):
        chunks = jnp.split(arr, indices)
        return treedef.unflatten([chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)])

    return rebuild


def tree_cast(tree: PyTree, dtype: Any) -> PyTree:
    return tree_map(lambda l: l.astype(dtype), tree)


def tree_isfinite(tree: PyTree) -> PyTree:
    return tree_reduce(jnp.logical_and, tree_map(lambda l: jnp.isfinite(l).all(), tree))


def tree_randn_like(tree: PyTree, key: Key) -> PyTree:

    leaves, treedef = tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))

    return treedef.unflatten(
        [jax.random.normal(k, l.shape, l.dtype) for k, l in zip(keys, leaves)]
    )  # type: ignore


def tree_func(f: Callable) -> Callable:
    @wraps(f)
    def tree_f(tree: PyTree, *args, **kwargs):
        f_ = partial(f, *args, **kwargs)
        return tree_map(f_, tree)

    return tree_f


tree_conj = tree_func(jnp.conj)
tree_real = tree_func(jnp.real)
tree_imag = tree_func(jnp.imag)
tree_shape = tree_func(jnp.shape)
tree_mean = tree_func(jnp.mean)
