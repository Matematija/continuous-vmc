from typing import Callable, Optional, Sequence
from functools import partial

import jax
from jax import numpy as jnp
from jax.tree_util import tree_map

from jax.numpy.fft import fftn, ifftn

from .utils.ad import grad
from .utils.stats import tau_int, circvar
from .utils.tree import tree_vdot
from .utils.chunk import vmap_chunked

from .hamiltonian import LocalEnergy, eloc_value_and_grad
from .qgt import QuantumGeometricTensor
from .utils import Array, PyTree, Ansatz, Scalar

Observable = Callable[[PyTree, Array], Array]


def correlation_func(_: PyTree, samples: Array) -> Array:

    """Compute the correlation function of a set of samples.

    Parameters
    ----------
    _ : PyTree
        Parameters placeholder. Unused parameter.
    samples : Array
        Samples to compute the correlation function of.

    Returns
    -------
    Array
        The correlation function of the samples.
    """

    n = jnp.stack([jnp.cos(samples), jnp.sin(samples)], axis=1)

    n_mean = n.mean(axis=0, keepdims=True)
    # n_mean, _ = mpi_allreduce_mean(n_mean)
    nc = n - n_mean

    ft = fftn(nc, s=nc.shape[2:])
    corr = ifftn(ft * ft.conj(), s=ft.shape[2:]).real

    return corr.sum(axis=1).mean(axis=0)


def kinetic_energy(eloc: LocalEnergy) -> Observable:

    """Compute the kinetic energy of a set of samples.

    Parameters
    ----------
    eloc : LocalEnergy
        Local energy object.

    Returns
    -------
    Callable
        Function that computes the kinetic energy of a set of samples.
    """

    kf = jax.vmap(eloc.kinetic_fn, in_axes=(None, 0))

    def ke_func(params: PyTree, samples: Array) -> Array:
        ke_mean = kf(params, samples).mean(axis=0)
        # ke_mean, _ = mpi_allreduce_mean(ke_mean)
        return ke_mean

    return ke_func


def potential_energy(eloc: LocalEnergy) -> Observable:

    """Compute the potential energy of a set of samples.

    Parameters
    ----------
    eloc : LocalEnergy
        Local energy object.

    Returns
    -------
    Callable
        Function that computes the potential energy of a set of samples.
    """

    pf = jax.vmap(eloc.potential_fn, in_axes=(None, 0))

    def pe_func(params: PyTree, samples: Array) -> Array:
        pe_mean = pf(params, samples).mean(axis=0)
        # pe_mean, _ = mpi_allreduce_mean(pe_mean)
        return pe_mean

    return pe_func


def magnetization(_: PyTree, samples: Array) -> Array:

    """Compute the magnetization of a set of samples, defined
    as the norm of the angular mean of samples/angles.

    Parameters
    ----------
    _ : PyTree
        Parameters placeholder. Unused parameter.
    samples : Array
        Samples to compute the magnetization of.

    Returns
    -------
    Array
        The magnetization of the samples.
    """

    samples = samples.reshape(samples.shape[0], -1)

    mx = jnp.cos(samples).mean(axis=1)
    my = jnp.sin(samples).mean(axis=1)

    # mx, token = mpi_allreduce_mean(mx)
    # my, _ = mpi_allreduce_mean(my, token=token)

    return jnp.hypot(mx, my).mean()


def average_directions(_: PyTree, samples: Array) -> Array:

    """Compute the angular mean direction of a set of samples/angles,
    defined as the normalized angular mean of samples/angles.

    Parameters
    ----------
    _ : PyTree
        Parameters placeholder. Unused parameter.
    samples : Array
        Samples to compute the average direction of.

    Returns
    -------
    Array
        The average direction of the samples.
    """

    shape = samples.shape[1:]
    samples = samples.reshape(samples.shape[0], -1)

    n = jnp.stack([jnp.cos(samples), jnp.sin(samples)], axis=-1).mean(axis=0)
    # n, _ = mpi_allreduce_mean(n)

    norms = jnp.linalg.norm(n, axis=-1, keepdims=True)
    n /= jnp.where(~jnp.isclose(norms, 0), norms, 1)

    return n.reshape(*shape, 2)


def direction_variances(_: PyTree, samples: Array) -> Array:

    """Compute the angular variances of the angular mean of a set of samples/angles.
    https://en.wikipedia.org/wiki/Directional_statistics#Measures_of_location_and_spread

    _ : PyTree
        Parameters placeholder. Unused parameter.

    samples : Array
        Samples to compute the angular variances of.

    Returns
    -------
    Array
        The angular variances of the samples.
    """
    return circvar(samples, axis=0)


def vorticity(scale: int, chunk_size: Optional[int] = None) -> Observable:

    """Compute the spatially-averaged vorticity of a set of samples,
    around a square of size `scale`.

    Parameters
    ----------
    scale : int
        Size of the square to compute the vorticity around.
    chunk_size : int, optional
        Chunk size for jax.vmap vectorization. Defaults to None.

    Returns
    -------
    Callable
        Function that computes the average vorticity of a set of samples.
    """

    assert scale >= 2, "Vorticity scale must be > 2 (single plaquette)"

    fx = jnp.zeros(shape=(scale, scale))
    fy = jnp.zeros(shape=(scale, scale))

    boundary = jnp.ones(scale).at[0].set(0.5).at[-1].set(0.5)

    fx = fx.at[0, :].set(-boundary)
    fx = fx.at[-1, :].set(boundary)
    fy = fy.at[:, 0].set(-boundary)
    fy = fy.at[:, -1].set(boundary)

    convolve = vmap_chunked(
        partial(jax.scipy.signal.convolve2d, mode="valid"), in_axes=(0, None), chunk_size=chunk_size
    )

    def vorticity_func(_: PyTree, samples: PyTree) -> Array:
        vol = samples.shape[1] * samples.shape[2]
        v = convolve(jnp.cos(samples), fx) + convolve(jnp.sin(samples), fy)
        return v.mean(axis=0) / vol

    return vorticity_func


def all_vorticities(dims: Sequence[int]) -> Sequence[Array]:
    """Compute the vorticity of a set of samples, for all possible scales.

    Parameters
    ----------
    dims : Sequence[int]
        Dimensions of the lattice.

    Returns
    -------
    Sequence[Array]
        The vorticity of the samples, for all possible scales.
    """

    fns = [vorticity(scale) for scale in range(2, min(*dims) + 1)]

    def vorticity_func(params: PyTree, samples: Array) -> Sequence[Array]:
        return tuple(f(params, samples) for f in fns)

    return vorticity_func


def mean_rotor_correlation(initial_samples: Array) -> Observable:
    """Compute the lattice-averaged rotor correlation of a set of samples,
    between two different times

    Parameters
    ----------
    initial_samples : Array
        Samples from the initial state.

    Returns
    -------
    Observable
        Function that computes the average rotor correlation
        between the initial state and a given state.
    """

    n0 = jnp.stack([jnp.cos(initial_samples), jnp.sin(initial_samples)], axis=-1)
    mean0 = jnp.mean(n0, axis=0, keepdims=True)
    # mean0, _ = mpi_allreduce_mean(mean0)
    nc0 = n0 - mean0

    del mean0, n0

    def mean_corr(_: PyTree, samples: Array) -> Array:

        n = jnp.stack([jnp.cos(samples), jnp.sin(samples)], axis=-1)
        n_mean = jnp.mean(n, axis=0, keepdims=True)
        # n_mean, token = mpi_allreduce_mean(n_mean)
        nc = n - n_mean

        c = jnp.sum(nc * nc0, axis=-1).mean()
        # c, _ = mpi_allreduce_mean(c, token=token)

        return c

    return mean_corr


def fidelity(
    logpsi: Ansatz, initial_params: PyTree, initial_samples: Array, chunk_size: Optional[int] = None
) -> Observable:
    """Compute the fidelity between the initial state and a given state.

    Parameters
    ----------
    logpsi : Ansatz
        The ansatz to use.
    initial_params : PyTree
        Parameters describing the initial state.
    initial_samples : Array
        Samples from the initial state.
    chunk_size : Optional[int], optional
        Chunk size for jax.vmap vectorization. Defaults to None.

    Returns
    -------
    Observable
        Function that computes the fidelity between the ansatz
        states described by `initial_params` and given `params`.
    """

    apply_fn = logpsi.apply if hasattr(logpsi, "apply") else logpsi
    batched_apply = vmap_chunked(apply_fn, in_axes=(None, 0), chunk_size=chunk_size)
    initial_logpsi_vals = batched_apply(initial_params, initial_samples)

    def f(params: PyTree, samples: Array) -> Scalar:

        term1 = batched_apply(initial_params, samples) - batched_apply(params, samples)
        term2 = batched_apply(params, initial_samples) - initial_logpsi_vals

        term1 = jax.nn.logsumexp(term1) - jnp.log(len(term1))
        term2 = jax.nn.logsumexp(term2) - jnp.log(len(term2))

        # term1, token = mpi_allreduce_mean(term1)
        # term2, _ = mpi_allreduce_mean(term2, token=token)

        return jnp.real(jnp.exp(term1 + term2))

    return f


def energy_variance(eloc: LocalEnergy, chunk_size: Optional[int] = None) -> Observable:

    """Compute the variance of the energy of a set of samples.

    Parameters
    ----------
    eloc : LocalEnergy
        The local energy observable.
    chunk_size : Optional[int], optional
        Chunk size for jax.vmap vectorization. Defaults to None.

    Returns
    -------
    Observable
        Function that computes the variance of the energy of a set of samples.
    """

    eloc_fn = vmap_chunked(eloc, in_axes=(None, 0), chunk_size=chunk_size)

    def var(params: PyTree, samples: Array) -> Scalar:
        eloc_vals = eloc_fn(params, samples)
        return jnp.mean(eloc_vals.real**2 + eloc_vals.imag**2)

    return var


def tdvp_error(logpsi: Ansatz, eloc: LocalEnergy, *args, **kwargs) -> Observable:

    qgt = QuantumGeometricTensor(logpsi, *args, **kwargs)

    def error(params: PyTree, samples: Array, *args) -> Scalar:

        _, g, Ec = eloc_value_and_grad(eloc, params, samples, chunk_size=eloc.chunk_size)

        theta_dot, _ = qgt.solve(params, samples, g, *args)
        corr = tree_vdot(g, theta_dot)

        v = jnp.mean(Ec.real**2 + Ec.imag**2)
        eps = jnp.finfo(v.dtype).eps

        return 1.0 - jnp.real(corr) / jnp.clip(v, eps)

    return error


########################################################################################################################

# Monte Carlo related stuff


def autocorr_time(
    observable: Callable,
    n_chains: int = 1,
    cutoff: Scalar = 0.05,
    max_lag: int = 1000,
    chunk_size: Optional[int] = None,
) -> Observable:

    obs_fn = vmap_chunked(observable, in_axes=(None, 0), chunk_size=chunk_size)

    def tau(params: PyTree, samples: Array) -> Array:
        Os = obs_fn(params, samples)
        return tau_int(Os, n_chains, cutoff, max_lag)

    return tau


def max_autocorr_time(
    observable: Callable,
    n_chains: int = 1,
    cutoff: Scalar = 0.05,
    max_lag: int = 1000,
    chunk_size: Optional[int] = None,
) -> Observable:

    tau_func = autocorr_time(observable, n_chains, cutoff, max_lag, chunk_size)

    def max_tau(params: PyTree, samples: Array) -> Scalar:
        taus = tau_func(params, samples)
        max_tau = jax.tree_reduce(jnp.maximum, tree_map(lambda l: jnp.max(jnp.abs(l)), taus))
        # max_tau, _ = mpi_allreduce_max(max_tau)
        return max_tau

    return max_tau


def grad_autocorr_time(
    logpsi: Ansatz,
    n_chains: int = 1,
    cutoff: Scalar = 0.05,
    max_lag: int = 1000,
    chunk_size: Optional[int] = None,
) -> Observable:

    grad_fn = grad(logpsi.apply, argnums=0)
    return autocorr_time(grad_fn, n_chains, cutoff, max_lag, chunk_size)


def max_grad_autocorr_time(
    logpsi: Ansatz,
    n_chains: int = 1,
    cutoff: Scalar = 0.05,
    max_lag: int = 1000,
    chunk_size: Optional[int] = None,
) -> Observable:

    grad_fn = grad(logpsi.apply, argnums=0)
    return max_autocorr_time(grad_fn, n_chains, cutoff, max_lag, chunk_size)
