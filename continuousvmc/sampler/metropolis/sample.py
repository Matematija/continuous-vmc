from typing import Callable, Sequence, Tuple, Optional
from functools import partial

import numpy as np

import jax
from jax import random
from jax.tree_util import Partial

from flax import struct

from .kernel import rwm_kernel
from .adaptation import warmup
from ..metric import IdentityMetric, Metric
from ..generic import (
    no_postprocessing,
    sample_chain,
    randn_init_fn,
    MCMCInfo,
    MCMCParams,
    MCMCState,
)

from ...utils.types import Array, Key, default_real


@struct.dataclass
class RWM:

    logp: Callable = struct.field(repr=False)
    params: MCMCParams
    initial_metric: Metric = struct.field(repr=False)

    def __call__(self, key: Key):
        return self.sample(key)

    def sample(self, key: Key):
        return rwm_sample(self.params, self.logp, self.initial_metric, key)


def RandomWalkMetropolis(
    log_prob: Callable,
    dims: Sequence[int],
    n_samples: int,
    n_chains: int,
    warmup: int,
    sweep: int = 1,
    *,
    adapt_step_size: bool = True,
    target_acc_rate: Scalar = 0.3,
    initial_step_size: Scalar = 0.1,
    step_size_bounds: Tuple[Scalar, Scalar] = (1e-8, 10.0),
    init_step_size_search: bool = True,
    adapt_metric: bool = True,
    diagonal_metric: bool = False,
    initial_metric: Optional[Metric] = None,
    init_fn: Optional[Callable] = None,
    postprocess_proposal: Optional[Callable] = None,
    angular: bool = False,
    **logp_kwargs,
) -> RWM:

    if initial_metric is None:
        initial_metric = IdentityMetric()

    logp = Partial(log_prob, **logp_kwargs)

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

    rwm_params = MCMCParams(
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
        initial_step_size=initial_step_size,
        init_step_size_search=init_step_size_search,
        init_fn=init_fn,
        postprocess_fn=postprocess_fn,
    )

    return RWM(logp, rwm_params, initial_metric)


def sample_rwm_chain(
    rwm_params: MCMCParams, logp: Callable, metric: Metric, key: Key
) -> Tuple[Array, MCMCInfo]:

    key1, key2, key3 = random.split(key, 3)
    dtype = default_real()

    init = rwm_params.init_fn(shape=rwm_params.dims, key=key1, dtype=dtype)
    state = MCMCState(
        x=init, accepted=True, acc_prob=0.0, step_size=rwm_params.initial_step_size, n_steps=0
    )

    metric_kwargs = {"circular": rwm_params.angular, "diagonal": rwm_params.diagonal_metric}
    state, metric = warmup(rwm_params, logp, metric, state, key2, metric_kwargs=metric_kwargs)
    kernel = rwm_kernel(rwm_params, logp, metric)

    samples = sample_chain(
        kernel, state, key3, n_samples=rwm_params.n_samples, sweep=rwm_params.sweep
    )

    info = MCMCInfo(
        acceptances=samples.accepted.mean(),
        metric=metric if rwm_params.adapt_metric else None,
        n_chains=1,
        step_size=state.step_size,
    )

    return samples.x, info


@partial(jax.jit, static_argnames="rwm_params")
def rwm_sample(
    rwm_params: MCMCParams, logp: Callable, initial_metric: Metric, key: Key
) -> Tuple[Array, MCMCInfo]:

    keys = random.split(key, rwm_params.n_chains)

    samples, info = jax.vmap(
        sample_rwm_chain,
        in_axes=(None, None, None, 0),
        out_axes=(0, 0),
        # chunk_size=hmc_params.chunk_size,
    )(rwm_params, logp, initial_metric, keys)

    info = info.replace(n_chains=rwm_params.n_chains)
    out_samples = samples.reshape(-1, *rwm_params.dims).squeeze()

    return out_samples, info
