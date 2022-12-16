from typing import Callable, Optional, Sequence, Union, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import Partial, tree_map

from flax import struct

from .utils import tree_conj, vmap_chunked
from .utils.ad import vjp, grad_and_diag_hess
from .utils.types import Scalar, Ansatz, Array, PyTree, tree_is_real, tree_is_complex, complex_dtype


@struct.dataclass
class LocalEnergy:
    def value_and_grad(self, params: PyTree, samples: Array):

        E_mean, grad_E, *_ = eloc_value_and_grad(self, params, samples, chunk_size=self.chunk_size)

        E_mean = jnp.real(E_mean)

        if tree_is_real(params):
            grad_E = tree_map(lambda l: 2 * jnp.real(l), grad_E)

        return E_mean, grad_E


@partial(jax.jit, static_argnames="chunk_size")
def eloc_value_and_grad(
    eloc: LocalEnergy, params: PyTree, samples: Array, chunk_size: Optional[int]
) -> Tuple[Scalar, PyTree, PyTree]:
    """Calculate the local energy and its gradient from model parameters and samples.

    Parameters
    ----------
    eloc : LocalEnergy
        The local energy instance.
    params : PyTree
        Model parameters.
    samples : Array
        Samples from the model. Expected shape is (n_samples, *lattice).
    chunk_size : int, optional
        Chunk size for the vmap. If None, no chunking is performed.

    Returns
    -------
    Tuple[Scalar, PyTree, PyTree]
        The local energy value, its gradient, and the auxiliary quantity: E - <E>.
    """

    eloc_batched = vmap_chunked(eloc, in_axes=(None, 0), chunk_size=chunk_size)
    E = eloc_batched(params, samples)

    E_mean = E.mean()
    Ec = E - E_mean

    batched_apply = jax.vmap(eloc.apply_fn, in_axes=(None, 0))
    _, back = vjp(lambda p: batched_apply(p, samples), params)

    (grad_E,) = tree_conj(back(jnp.conj(Ec) / samples.shape[0]))

    return E_mean, grad_E, Ec


################################################################################


def _make_kinetic_fn(logpsi: Ansatz, full_hessian: bool = False) -> Callable:

    apply_fn = logpsi.apply if hasattr(logpsi, "apply") else logpsi

    if full_hessian:

        def flat_eval(params, flat_sample):
            sample = flat_sample.reshape(*logpsi.dims)
            return apply_fn(params, sample)

        @Partial
        def kinetic_fn(params, sample):

            holomorphic = tree_is_complex(params)

            grad_fn = jax.grad(flat_eval, argnums=1, holomorphic=holomorphic)
            hess_fn = jax.hessian(flat_eval, argnums=1, holomorphic=holomorphic)

            dtype = complex_dtype(sample.dtype)
            sample = jnp.asarray(sample, dtype=dtype)

            grads = grad_fn(params, sample.ravel())
            hess = hess_fn(params, sample.ravel())

            return -jnp.diag(hess).sum() - jnp.sum(grads**2)

    else:

        grad_and_diag_hess_fn = grad_and_diag_hess(apply_fn, argnums=1)

        @Partial
        def kinetic_fn(params, sample):
            grads, diag_hess = grad_and_diag_hess_fn(params, sample)
            return -jnp.sum(diag_hess) - jnp.sum(grads**2)

    return kinetic_fn


def _make_potential_fn(
    dims: Sequence[int], pbc: bool, twist: Optional[Sequence[int]] = None
) -> Callable:

    dim = len(dims)

    if twist is None:
        twist = (0,) * dim
    elif isinstance(twist, int):
        twist = (twist,) + (0,) * (dim - 1)
    elif len(twist) != dim:
        twist = tuple(twist) + (0,) * (dim - len(twist))
    else:
        twist = tuple(twist)

    if pbc:
        phis = tuple(2 * jnp.pi * n / L for n, L in zip(twist, dims))

    if pbc:

        @Partial
        def potential_fn(_, sample):

            out = 0.0

            for i, phi in enumerate(phis):
                d = sample - jnp.roll(sample, shift=1, axis=i)
                out -= jnp.cos(d + phi).sum()

            return out

    else:

        @Partial
        def potential_fn(_, sample):

            out = 0.0

            for i in range(dim):
                d = jnp.diff(sample, axis=i)
                out -= jnp.cos(d).sum()

            return out

    return potential_fn


def _add_external_field_term(potential_fn: Callable, h: Scalar) -> Callable:

    h = jnp.asarray(h)

    @Partial
    def external_field_term(_, sample):
        val = potential_fn(_, sample)
        extra = jnp.sum(h * jnp.cos(sample))
        return val + extra

    return external_field_term


############################################################################


@struct.dataclass
class QuantumRotorModel(LocalEnergy):

    dims: tuple
    g: Scalar
    apply_fn: Callable[[PyTree, Array], Array] = struct.field(repr=False)
    kinetic_fn: Callable[[PyTree, Array], Array] = struct.field(repr=False)
    potential_fn: Callable[[Array], Array] = struct.field(repr=False)
    J: Scalar = 1.0
    pbc: bool = True
    chunk_size: Optional[int] = None

    def __call__(self, params: PyTree, sample: Array) -> Array:
        return _quantum_rotor_model(self, params, sample)

    def to_dict(self):
        return dict(dims=self.dims, g=self.g, J=self.J, pbc=self.pbc, chunk_size=self.chunk_size)


def _quantum_rotor_model(eloc: QuantumRotorModel, params: PyTree, sample: Array) -> Array:
    kinetic_term = eloc.kinetic_fn(params, sample)
    potential_term = eloc.potential_fn(params, sample)
    return (0.5 * eloc.g * eloc.J) * kinetic_term + eloc.J * potential_term


def QuantumRotors(
    logpsi: Ansatz,
    g: Scalar,
    J: Scalar = 1.0,
    h: Optional[Union[Scalar, Array]] = None,
    *,
    pbc: bool = False,
    twist: Optional[Union[int, Sequence[int]]] = None,
    full_hessian: bool = False,
    chunk_size: Optional[int] = None,
) -> QuantumRotorModel:

    """Quantum rotor model.

    Parameters
    ----------
    logpsi : Ansatz
        The ansatz. Should have a `apply` method that takes a parameter tree and
        a sample and returns the log of the wavefunction.
    g : Scalar
        The `g` coupling constant for the Quantum Rotor Model kinetic term.
    J : Scalar, optional
        The Quantum Rotor Model energy scale, by default 1.0
    h : Union[Scalar, Array], optional
        An external local or global field for the Quantum Rotor Model, by default None.
    pbc : bool, optional
        Whether to use periodic boundary conditions, by default False.
    twist : Union[int, Sequence[int]], optional
        Twist in the boundary conditions, incorporating the effects of an external vector
        potential by default None. WARNING: This feature is not thoroughly tested!
    full_hessian : bool, optional
        Whether to construct the full Hessian or the row-by-row diagonal Hessian when
        calculating the Laplacian, by default False.
    chunk_size : int, optional
        The chunk size for future gradient computation on many samples, by default None.

    Returns
    -------
    QuantumRotorModel
        The quantum rotor model local energy.

    Raises
    ------
    ValueError
        If J is not positive.
    ValueError
        If the ansatz does not describe a 1D or a 2D system.
    """

    if J <= 0:
        raise ValueError(f"J must be positive! got J={J}")

    dims = tuple(logpsi.dims)

    if len(dims) not in [1, 2]:
        raise ValueError(f"Dimensions not 1 or 2 not supported yet! Got dim={len(dims)}")

    apply_fn = Partial(logpsi.apply)

    if g != 0:
        kinetic_fn = _make_kinetic_fn(logpsi, full_hessian=full_hessian)
    else:
        kinetic_fn = Partial(lambda *_, **__: 0.0)

    potential_fn = _make_potential_fn(dims, pbc=pbc, twist=twist)

    if h is not None:
        potential_fn = _add_external_field_term(potential_fn, h)

    return QuantumRotorModel(
        dims=tuple(logpsi.dims),
        g=Scalar(g),
        apply_fn=apply_fn,
        kinetic_fn=kinetic_fn,
        potential_fn=potential_fn,
        J=Scalar(J),
        pbc=bool(pbc),
        chunk_size=chunk_size,
    )
