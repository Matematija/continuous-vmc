from typing import Callable, Optional, Sequence, Tuple
from functools import partial

import numpy as np

import jax
from jax import random
from jax.tree_util import Partial

from flax import struct

from .kernel import hmc_kernel, HMCParams
from .hamiltonian import Hamiltonian
from .adaptation import warmup
from ..metric import Metric, IdentityMetric
from ..generic import sample_chain, no_postprocessing, randn_init_fn, MCMCInfo, MCMCState

from ...utils.types import Key, Scalar, default_real


@struct.dataclass
class HMC:

    params: HMCParams
    initial_hamiltonian: Hamiltonian

    def __call__(self, key: Key):
        return self.sample(key)

    def sample(self, key: Key):
        return hmc_sample(self.params, self.initial_hamiltonian, key)


def HamiltonianMonteCarlo(
    log_prob: Callable,
    dims: Sequence[int],
    n_samples: int,
    n_chains: int,
    warmup: int,
    n_leaps: int,
    sweep: int = 1,
    *,
    adapt_step_size: bool = True,
    target_acc_rate: Scalar = 0.8,
    initial_step_size: Scalar = 0.1,
    step_size_bounds: Tuple[Scalar, Scalar] = (1e-8, 10.0),
    init_step_size_search: bool = True,
    adapt_metric: bool = True,
    diagonal_metric: bool = False,
    jitter: Scalar = 0.2,
    initial_metric: Optional[Metric] = None,
    init_fn: Optional[Callable] = None,
    postprocess_proposal: Optional[Callable] = None,
    angular: bool = False,
    chunk_size: Optional[int] = None,
    **log_prob_kwargs,
) -> HMC:

    if initial_metric is None:
        initial_metric = IdentityMetric()

    logp = Partial(log_prob, **log_prob_kwargs)
    initial_hamiltonian = Hamiltonian(logp, initial_metric)

    if init_fn is None:
        init_fn = randn_init_fn
    elif callable(init_fn):
        if not isinstance(init_fn, Partial):
            init_fn = Partial(init_fn)
    else:
        raise TypeError(f"`init_fn` must be a callable or None, got {init_fn}")

    if postprocess_proposal is None:
        postprocess_fn = no_postprocessing
    elif callable(postprocess_proposal):
        if not isinstance(postprocess_proposal, Partial):
            postprocess_fn = Partial(postprocess_proposal)
        else:
            postprocess_fn = postprocess_proposal

    else:
        raise TypeError(f"`postprocess_proposal` must be callable, got {postprocess_proposal}")

    log_step_size_bounds = tuple(np.log10(x) for x in step_size_bounds)

    hmc_params = HMCParams(
        dims=dims,
        n_samples=n_samples,
        n_chains=n_chains,
        sweep=sweep,
        warmup=warmup,
        adapt_step_size=adapt_step_size,
        adapt_metric=adapt_metric,
        target_acc_rate=target_acc_rate,
        log_step_size_bounds=log_step_size_bounds,
        angular=angular,
        diagonal_metric=diagonal_metric,
        n_leaps=n_leaps,
        initial_step_size=initial_step_size,
        init_step_size_search=init_step_size_search,
        init_fn=init_fn,
        postprocess_fn=postprocess_fn,
        jitter=jitter,
        chunk_size=chunk_size,
    )

    return HMC(hmc_params, initial_hamiltonian)


def sample_hmc_chain(hmc_params: HMCParams, initial_h: Hamiltonian, key: Key):

    key1, key2, key3 = random.split(key, 3)
    dtype = default_real()

    init = hmc_params.init_fn(shape=hmc_params.dims, key=key1, dtype=dtype)
    state = MCMCState(
        x=init, accepted=True, acc_prob=1.0, step_size=hmc_params.initial_step_size, n_steps=0
    )

    metric_kwargs = {"circular": hmc_params.angular, "diagonal": hmc_params.diagonal_metric}

    state, h = warmup(hmc_params, initial_h, state, key2, metric_kwargs=metric_kwargs)

    kernel = hmc_kernel(hmc_params, h)
    samples = sample_chain(kernel, state, key3, hmc_params.n_samples, hmc_params.sweep)

    info = MCMCInfo(samples.accepted.mean(), h.metric, 1, samples.step_size[-1], hmc_params.n_leaps)

    return samples.x, info


@partial(jax.jit, static_argnames="hmc_params")
def hmc_sample(hmc_params: HMCParams, initial_h: Hamiltonian, key: Key):

    keys = random.split(key, hmc_params.n_chains)

    samples, info = jax.vmap(
        sample_hmc_chain,
        in_axes=(None, None, 0),
        out_axes=(0, 0),
        # chunk_size=hmc_params.chunk_size,
    )(hmc_params, initial_h, keys)

    info = info.replace(n_chains=hmc_params.n_chains)
    out_samples = samples.reshape(-1, *hmc_params.dims).squeeze()

    return out_samples, info
