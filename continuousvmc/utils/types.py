from typing import Union, Callable
import jax, chex
import flax.linen as nn

####################################################################################################

Key = chex.PRNGKey
PyTree = chex.ArrayTree
Array = Union[chex.Array, chex.ArrayNumpy, chex.ArrayBatched]
Scalar = Union[chex.Scalar, chex.Numeric]
DType = chex.ArrayDType
Token = jax.core.Token

Ansatz = Union[nn.Module, Callable[[PyTree, Array], Scalar]]

####################################################################################################

import numpy as np

import jax
from jax import numpy as jnp
from jax.tree_util import tree_leaves


def is_complex_dtype(dtype: DType) -> bool:
    return jnp.issubdtype(dtype, jnp.complexfloating)


def is_real_dtype(dtype: DType) -> bool:
    return jnp.issubdtype(dtype, jnp.floating)


def is_complex(arr: Array) -> bool:
    return is_complex_dtype(arr.dtype)


def is_real(arr: Array) -> bool:
    return is_real_dtype(arr.dtype)


def tree_is_complex(tree: PyTree) -> bool:
    return any(is_complex(l) for l in tree_leaves(tree))


def tree_is_real(tree: PyTree) -> bool:
    return all(is_real(l) for l in tree_leaves(tree))


def tree_common_dtype(tree: PyTree) -> DType:
    return np.common_type(*tree_leaves(tree))


def real_dtype(dtype: DType) -> DType:

    if is_complex_dtype(dtype):
        if dtype == np.dtype("complex64"):
            return np.dtype("float32")
        elif dtype == np.dtype("complex128"):
            return np.dtype("float64")
        else:
            raise TypeError(f"Unknown complex dtype {dtype}.")
    else:
        return np.dtype(dtype)


def complex_dtype(dtype: DType) -> DType:

    if is_real_dtype(dtype):
        if dtype == np.dtype("float32"):
            return np.dtype("complex64")
        elif dtype == np.dtype("float64"):
            return np.dtype("complex128")
        else:
            raise TypeError(f"Unknown real dtype {dtype}.")
    else:
        return np.dtype(dtype)


def to_complex(arr: Array) -> Array:

    if is_real(arr):
        new_dtype = complex_dtype(arr.dtype)
        return arr.astype(new_dtype)
    else:
        return arr


def default_real() -> DType:
    return np.dtype("float64" if jax.config.x64_enabled else "float32")  # type: ignore


def default_complex() -> DType:
    return complex_dtype(default_real())
