from typing import Callable, Tuple, Union, Sequence

import numpy as np

import jax
from jax.tree_util import tree_map
from jax import linear_util as lu
from jax.api_util import argnums_partial, flatten_fun_nokwargs
from jax.interpreters.ad import jvp as ad_jvp

from jax import numpy as jnp

from jax.tree_util import tree_flatten, tree_unflatten, tree_reduce

from .grad import grad
from ..tree import _ravel_list, tree_size, basis_of
from ..types import PyTree


def _partition_into_leaves(x, shapes):
    indices = np.cumsum([np.prod(s) for s in shapes])[:-1]
    chunks = jnp.split(x, indices)
    return tuple(chunk.reshape(shape) for chunk, shape in zip(chunks, shapes))


def grad_and_diag_hess(f: Callable, argnums: Union[int, Sequence[int]] = 0):

    grad_fn = grad(f, argnums=argnums)

    def grad_and_diag_hess_fn(*args, **kwargs) -> Tuple[PyTree, PyTree]:

        g = lu.wrap_init(grad_fn, kwargs)

        g_partial, dyn_args = argnums_partial(g, argnums, args, require_static_args_hashable=False)

        primals_flat, treedef = tree_flatten(dyn_args)
        flat_fun, out_treedef = flatten_fun_nokwargs(g_partial, treedef)

        def vmap_fun(tangents, i):

            tangents_flat, _ = tree_flatten(tangents)
            val, fwd = ad_jvp(flat_fun).call_wrapped(primals_flat, tangents_flat)

            fwd_val = _ravel_list(*fwd)[i]  # TODO: Is there a better way of doing this?
            return val, fwd_val

        basis = basis_of(dyn_args)
        ixs = jnp.arange(tree_size(dyn_args))

        grad_flat, diag_hess_flat = jax.vmap(vmap_fun, in_axes=(0, 0), out_axes=(None, 0))(
            basis, ixs
        )
        # Maybe make this a scan? Or a chunked vmap?

        diag_hess_flat = _partition_into_leaves(
            diag_hess_flat, shapes=[l.shape for l in primals_flat]
        )

        grad = tree_unflatten(out_treedef(), grad_flat)
        diag_hess = tree_unflatten(out_treedef(), diag_hess_flat)

        return grad, diag_hess

    return grad_and_diag_hess_fn


def diag_hess(f: Callable, argnums: Union[int, Sequence[int]] = 0):

    grad_and_diag_hess_fn = grad_and_diag_hess(f, argnums=argnums)

    def diag_hess_fn(*args, **kwargs) -> PyTree:
        _, diag_hess = grad_and_diag_hess_fn(*args, **kwargs)
        return diag_hess

    return diag_hess_fn


def laplacian(f: Callable, argnums: Union[int, Sequence[int]] = 0):

    diag_hess_fn = diag_hess(f, argnums=argnums)

    def laplacian_fn(*args, **kwargs) -> PyTree:
        diag_hess = diag_hess_fn(*args, **kwargs)
        return tree_reduce(jnp.add, tree_map(jnp.sum, diag_hess))

    return laplacian_fn
