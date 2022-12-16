from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

import jax
from jax import numpy as jnp
from jax.tree_util import Partial

from flax import struct

from .sample import HMC
from .kernel import HMCParams
from .hamiltonian import Hamiltonian
from ..metric import IdentityMetric
from ..generic import no_postprocessing, center_proposal, randn_init_fn, angular_init_fn

from ...utils import eval_observables
from ...utils.types import Key, PyTree, Ansatz, Scalar


@struct.dataclass
class VHMC:

    log_prob_fun: Callable = struct.field(repr=False)
    params: HMCParams
    eval_observables: Callable = struct.field(repr=False, pytree_node=False)
    project: bool = False

    def __call__(self, params: PyTree, key: Key):
        return self.sample(params, key)

    def sample(self, params: PyTree, key: Key):
        return _vmc_hmc_sample(
            self.params,
            self.log_prob_fun,
            self.eval_observables,
            self.project,
            params,
            key,
        )

    def to_dict(self):
        return {k: v for k, v in self.params.__dict__.items() if not callable(v)}


def VariationalHMC(
    logpsi: Ansatz,
    n_samples: int,
    n_chains: int,
    warmup: int,
    n_leaps: int,
    sweep: int = 1,
    *,
    dims: Optional[Sequence[int]] = None,
    adapt_step_size: bool = True,
    step_size_bounds: Tuple[Scalar, Scalar] = (1e-8, 10.0),
    adapt_metric: bool = True,
    diagonal_metric: bool = True,
    jitter: Scalar = 0.2,
    target_acc_rate: Scalar = 0.65,
    initial_step_size: Scalar = 0.1,
    init_step_size_search: bool = True,
    angular: bool = False,
    augmented: bool = False,
    observables: Optional[Sequence[Callable]] = None,
    chunk_size: Optional[int] = None,
) -> VHMC:

    if augmented:

        angular = False

        @Partial
        def logp(params, xy):
            thetas = jnp.arctan2(xy[1], xy[0])
            return -0.5 * jnp.sum(xy**2) + logpsi.log_prob(params, thetas)

        dims = (2,) + logpsi.dims if dims is None else tuple(dims)
    else:
        logp = Partial(logpsi.log_prob)
        dims = logpsi.dims if dims is None else tuple(dims)

    if angular:
        postprocess_proposal = center_proposal
        init_fn = angular_init_fn
    else:
        postprocess_proposal = no_postprocessing
        init_fn = randn_init_fn

    log_step_size_bounds = tuple(np.log(x) for x in step_size_bounds)

    hmc_params = HMCParams(
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
        n_leaps=n_leaps,
        initial_step_size=initial_step_size,
        init_fn=init_fn,
        postprocess_fn=postprocess_proposal,
        jitter=jitter,
        chunk_size=chunk_size,
    )

    if observables is not None and not isinstance(observables, tuple):

        if hasattr(observables, "__iter__"):
            observables = tuple(observables)
        else:
            observables = (observables,)

    eval_obs = partial(eval_observables, observables)

    return VHMC(logp, hmc_params, eval_obs, bool(augmented))


@partial(jax.jit, static_argnames=["hmc_params", "eval_obs", "project"])
def _vmc_hmc_sample(
    hmc_params: HMCParams,
    log_prob_fn: Partial,
    eval_obs: Callable,
    project: bool,
    params: PyTree,
    key: Key,
):

    logp = Partial(log_prob_fn, params)

    metric = IdentityMetric()
    initial_hamiltonian = Hamiltonian(logp, metric)
    hmc = HMC(hmc_params, initial_hamiltonian)
    samples, info = hmc.sample(key)

    if project:
        samples = jnp.arctan2(samples[:, 1], samples[:, 0])

    observables = eval_obs(params, samples)

    return samples, observables, info
