from typing import Callable, Optional

from jax import numpy as jnp
from jax.tree_util import tree_map

from flax import struct

from .ansatz import canonicalize_ansatz
from .hamiltonian import LocalEnergy, eloc_value_and_grad
from .qgt import QuantumGeometricTensor
from .utils import eval_shape
from .utils.types import Ansatz, Key, PyTree, Scalar, Array, tree_is_real


@struct.dataclass
class VMCInfo:
    energy: Optional[Scalar] = None
    observables: Optional[PyTree] = struct.field(repr=False, default=None)
    sampler_info: Optional[PyTree] = struct.field(repr=False, default=None)
    solver_info: Optional[PyTree] = struct.field(repr=False, default=None)


def ParameterDerivative(
    logpsi: Ansatz,
    eloc: LocalEnergy,
    sampler: Callable,
    eps: Scalar,
    *,
    real_time: bool = True,
    return_samples: bool = False,
    solver: str = "shift",
    **solver_kwargs
) -> Callable:

    qgt = QuantumGeometricTensor(
        logpsi, solver=solver, eps=eps, chunk_size=eloc.chunk_size, **solver_kwargs
    )

    apply_fn = canonicalize_ansatz(logpsi)

    def derivative(
        params: PyTree, key: Key, init_samples: Optional[Array] = None, x0: Optional[PyTree] = None
    ):

        # key, _ = mpi_scatter_keys(key)

        samples, observables, sampler_info = sampler(params, key, init_samples=init_samples)
        out_shape = eval_shape(apply_fn, params, samples[0])

        if tree_is_real(out_shape) and real_time:
            raise RuntimeError("Cannot do time evolution with purely real wavefunctions.")

        E, rhs, Ec = eloc_value_and_grad(eloc, params, samples, chunk_size=eloc.chunk_size)

        if real_time:
            rhs = tree_map(lambda l: -1j * l, rhs)

        if tree_is_real(params):
            rhs = tree_map(jnp.real, rhs)

        grads, solver_info = qgt.solve(params, samples, rhs, Ec, x0=x0)

        info = VMCInfo(
            energy=jnp.real(E),
            observables=observables,
            sampler_info=sampler_info,
            solver_info=solver_info,
        )

        if return_samples:
            return grads, samples, info
        else:
            return grads, info

    return derivative
