from typing import Optional, Union, Sequence, Tuple
from functools import partial

import jax
from jax import lax
from jax import numpy as jnp
from jax.numpy import fft
from jax.tree_util import tree_map

from .chunk import vmap_chunked
from .types import Array, PyTree, Scalar, Key


def circvar(samples: Array, axis: Optional[Union[int, Sequence[int]]] = None) -> Array:

    cos_mean = jnp.cos(samples).mean(axis=axis)
    sin_mean = jnp.sin(samples).mean(axis=axis)

    R = jnp.minimum(1, jnp.hypot(cos_mean, sin_mean))
    return -2 * jnp.log(R)


def circcov(samples: Array, batch_axis: int = 0, ddof: int = 1) -> Array:

    samples = jnp.swapaxes(samples, batch_axis, 0)

    if samples.ndim == 1:
        samples = jnp.expand_dims(samples, axis=1)

    n_samples = samples.shape[0]
    shape = samples.shape[1:]

    flat_samples = samples.reshape(n_samples, -1)

    cos_mean = jnp.cos(flat_samples).mean(axis=0, keepdims=True)
    sin_mean = jnp.sin(flat_samples).mean(axis=0, keepdims=True)
    mean = jnp.arctan2(sin_mean, cos_mean)

    sines = jnp.sin(samples - mean) / jnp.sqrt(samples.shape[0] - ddof)
    cov = sines.T.conj() @ sines

    return cov.reshape(*shape, *shape).squeeze() / (jnp.sqrt(2) - 1.0)


@partial(jax.jit, static_argnames="n_bootstraps")
def bootstrap_error(data: Array, key: Key, n_bootstraps: int = None):

    # data.shape = (*batch, n_samples)
    n_samples = data.shape[-1]

    if n_bootstraps is None:
        n_bootstraps = n_samples

    idxs = jax.random.choice(key, n_samples, shape=(n_bootstraps, n_samples), replace=True)
    resampled = jnp.take(data, idxs, axis=-1).mean(axis=-1)
    v = jnp.mean(resampled**2, axis=-1) - jnp.mean(resampled, axis=-1) ** 2

    return jnp.sqrt(v)


@jax.jit
def auto_corr(obs: Array) -> Array:

    obs = obs - obs.mean()
    N = len(obs)
    fvi = fft.fft(obs, n=2 * N)

    # G is the full autocorrelation curve

    G = jnp.real(fft.ifft(fvi * jnp.conjugate(fvi))[:N])
    G /= N - jnp.arange(N)  # type: ignore
    G = lax.cond(~jnp.isclose(G[0], 0.0), lambda G: G / G[0], lambda G: G, G)

    return G


def _tau_int(obs: Array, cutoff: Scalar = 0.05, max_lag: int = 1000) -> Tuple[Array, Array]:

    autocorr = auto_corr(obs)

    cutoff_ind = jnp.argmax(jnp.abs(autocorr) < cutoff)
    cutoff_ind = lax.cond(cutoff_ind == 0, lambda _: len(autocorr), lambda _: cutoff_ind, None)
    cutoff_ind = jnp.minimum(cutoff_ind, max_lag)

    where = jnp.arange(len(autocorr)) < cutoff_ind

    tau = 0.5 + autocorr.sum(where=where)
    masked_autocorr = jnp.where(where, autocorr, 0)

    return tau, masked_autocorr  # type: ignore


@partial(jax.jit, static_argnames=["n_chains", "chunk_size"])
def tau_int(
    obs: PyTree,
    n_chains: int = 1,
    cutoff: Scalar = 0.05,
    max_lag: int = 1000,
    chunk_size: Optional[int] = None,
) -> PyTree:

    tau_func_flat = lambda arr: _tau_int(arr, cutoff, max_lag)[0]

    tau_func = vmap_chunked(jax.vmap(tau_func_flat, in_axes=1), in_axes=0, chunk_size=chunk_size)

    def map_func(leaf):

        n_samples, *shape = leaf.shape
        samples_per_chain = n_samples // n_chains

        taus_flat = tau_func(leaf.reshape(n_chains, samples_per_chain, -1))

        return taus_flat.reshape(n_chains, *shape).squeeze()

    return tree_map(map_func, obs)


@partial(jax.jit, static_argnames="max_lag_scan")
def tau_ints(obs: Array, max_lag_scan: int = 800) -> PyTree:
    ts = jnp.arange(max_lag_scan)
    return jax.vmap(lambda i: _tau_int(obs, cutoff=0, max_lag=i)[0])(ts)
