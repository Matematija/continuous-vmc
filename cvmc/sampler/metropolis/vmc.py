from typing import Callable, Optional, Sequence, Tuple
from functools import partial

import numpy as np

import jax
from jax import numpy as jnp
from jax.tree_util import Partial

from flax import struct

from .sample import RWM
from ..metric import IdentityMetric
from ..generic import center_proposal, no_postprocessing, randn_init_fn, angular_init_fn, MCMCParams

from ...utils import eval_observables
from ...utils.types import Key, Ansatz, PyTree, Scalar


@struct.dataclass
class VRWM:

    log_prob: Callable = struct.field(repr=False)
    params: MCMCParams
    eval_observables: Callable = struct.field(repr=False, pytree_node=False)

    def __call__(self, params: PyTree, key: Key):
        return self.sample(params, key)

    def sample(self, params: PyTree, key: Key):
        return _vmc_rwm_sample(self.params, self.log_prob, self.eval_observables, params, key)


def VariationalMetropolis(
    logpsi: Ansatz,
    n_samples: int,
    n_chains: int,
    warmup: int,
    sweep: int = 1,
    *,
    adapt_step_size: bool = True,
    step_size_bounds: Tuple[Scalar, Scalar] = (1e-8, 10.0),
    adapt_metric: bool = True,
    diagonal_metric: bool = True,
    target_acc_rate: Scalar = 0.65,
    initial_step_size: Scalar = 0.1,
    init_step_size_search: bool = True,
    angular: bool = False,
    observables: Optional[Sequence[Callable]] = None,
) -> VRWM:

    if hasattr(logpsi, "log_prob"):
        log_prob = logpsi.log_prob
    else:
        log_prob = lambda params, x: 2 * jnp.real(logpsi(params, x))

    if not angular:
        postprocess_proposal = no_postprocessing
        init_fn = randn_init_fn
    else:
        postprocess_proposal = center_proposal
        init_fn = angular_init_fn

    dims = logpsi.dims if dims is None else tuple(dims)
    log_step_size_bounds = tuple(np.log(x) for x in step_size_bounds)

    rwm_params = MCMCParams(
        dims=dims,
        n_samples=n_samples,
        n_chains=n_chains,
        warmup=warmup,
        sweep=sweep,
        adapt_step_size=adapt_step_size,
        log_step_size_bounds=log_step_size_bounds,
        adapt_metric=adapt_metric,
        init_step_size_search=init_step_size_search,
        target_acc_rate=target_acc_rate,
        angular=angular,
        diagonal_metric=diagonal_metric,
        initial_step_size=initial_step_size,
        init_fn=init_fn,
        postprocess_fn=postprocess_proposal,
        chunk_size=None,
    )

    if observables is not None and not isinstance(observables, tuple):

        if hasattr(observables, "__iter__"):
            observables = tuple(observables)
        else:
            observables = (observables,)

    eval_obs = partial(eval_observables, observables)

    return VRWM(Partial(log_prob), rwm_params, eval_obs)


@partial(jax.jit, static_argnames=["rwm_params", "eval_obs"])
def _vmc_rwm_sample(
    rwm_params: MCMCParams, logp: Callable, eval_obs: Callable, params: PyTree, key: Key
):

    logp = Partial(logp, params)
    metric = IdentityMetric()

    samples, info = RWM(logp, rwm_params, metric).sample(key)
    observables = eval_obs(params, samples)

    return samples, observables, info
