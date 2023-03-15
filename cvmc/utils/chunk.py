from typing import Callable, Optional, Sequence, Union

import jax
from jax import lax
from jax import numpy as jnp

from jax.tree_util import tree_leaves, tree_map
from jax import linear_util as lu
from jax.api_util import argnums_partial  # type: ignore

from .types import Array
from .tree import tree_func


@tree_func
def chunk(x: Array, chunk_size: int = 1) -> Array:
    return x.reshape(-1, chunk_size, *x.shape[1:])


@tree_func
def unchunk(x: Array):
    return x.reshape(-1, *x.shape[2:])


def _get_chunks(x, n_bulk, chunk_size):

    bulk = tree_map(
        lambda l: lax.dynamic_slice_in_dim(l, start_index=0, slice_size=n_bulk, axis=0), x
    )

    return chunk(bulk, chunk_size=chunk_size)


def _get_rest(x, n_bulk, n_rest):
    return tree_map(
        lambda l: lax.dynamic_slice_in_dim(l, start_index=n_bulk, slice_size=n_rest, axis=0), x
    )


def map_over_chunks(fun, argnums=0):

    if isinstance(argnums, int):
        argnums = (argnums,)

    def mapped(*args, **kwargs):

        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args, require_static_args_hashable=False)

        return lax.map(lambda x: f_partial.call_wrapped(*x), dyn_args)  # type: ignore

    return mapped


def _chunk_vmapped_function(vmapped_fun, chunk_size, argnums):

    if chunk_size is None:
        return vmapped_fun

    def out_fun(*args, **kwargs):

        f = lu.wrap_init(vmapped_fun, kwargs)

        f_partial, dyn_args = argnums_partial(f, argnums, args, require_static_args_hashable=False)

        axis_len = tree_leaves(dyn_args)[0].shape[0]
        n_chunks, n_rest = divmod(axis_len, chunk_size)
        n_bulk = chunk_size * n_chunks

        bulk = _get_chunks(dyn_args, n_bulk, chunk_size)

        y = unchunk(lax.map(lambda x: f_partial.call_wrapped(*x), bulk))  # type: ignore

        if n_rest > 0:
            rest = _get_rest(dyn_args, n_bulk, n_rest)
            y_rest = f_partial.call_wrapped(*rest)  # type: ignore
            y = tree_map(lambda y1, y2: jnp.concatenate((y1, y2), axis=0), y, y_rest)

        return y

    return out_fun


def vmap_chunked(
    f: Callable, in_axes: Union[int, Sequence[int]] = 0, *, chunk_size: Optional[int] = None
) -> Callable:

    """A wrapper around jax.vmap that allows to chunk the input
    array using jax.lax.scan over leading axes for improved memory efficiency.

    Parameters
    ----------
    f : Callable
        (The original docstring of jax.vmap)
        Function to be mapped over additional axes.
    in_axes : Union[int, Sequence[int]], optional
        (The original docstring of jax.vmap)
        (tuple/list/dict) thereof specifying which input array axes to map over.

        If each positional argument to ``fun`` is an array, then ``in_axes`` can
        be an integer, a None, or a tuple of integers and Nones with length equal
        to the number of positional arguments to ``fun``. An integer or ``None``
        indicates which array axis to map over for all arguments (with ``None``
        indicating not to map any axis), and a tuple indicates which axis to map
        for each corresponding positional argument. Axis integers must be in the
        range ``[-ndim, ndim)`` for each array, where ``ndim`` is the number of
        dimensions (axes) of the corresponding input array.

        If the positional arguments to ``fun`` are container (pytree) types, the
        corresponding element of ``in_axes`` can itself be a matching container,
        so that distinct array axes can be mapped for different container
        elements. ``in_axes`` must be a container tree prefix of the positional
        argument tuple passed to ``fun``. See this link for more detail:
        https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees

        Either ``axis_size`` must be provided explicitly, or at least one
        positional argument must have ``in_axes`` not None. The sizes of the
        mapped input axes for all mapped positional arguments must all be equal.

        Arguments passed as keywords are always mapped over their leading axis
        (i.e. axis index 0).

        See below for examples.

    chunk_size : int, optional
        The size of the chunks to be used for the leading axis. If `None` no chunking is performed.

    Returns
    -------
    Callable
        _description_

    Raises
    ------
    NotImplementedError
        If any of the `in_axes` is not `0` or `None`. Chunking is only supported
        for leading axes right now.
    """

    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    if not set(in_axes).issubset((0, None)):
        raise NotImplementedError("Only in_axes 0/None are currently supported")

    argnums = tuple(i for i, ix in enumerate(in_axes) if ix is not None)
    vmapped_fun = jax.vmap(f, in_axes=in_axes)

    return _chunk_vmapped_function(vmapped_fun, chunk_size, argnums)
