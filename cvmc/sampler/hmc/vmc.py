from functools import partial
from warnings import warn
from typing import Callable, Optional, Sequence, Tuple, Any

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
from ...utils.types import Key, PyTree, Ansatz, Scalar, Array


@struct.dataclass
class VHMC:

    log_prob_fun: Callable = struct.field(repr=False)
    params: HMCParams
    eval_observables: Callable = struct.field(repr=False, pytree_node=False)

    def __call__(self, params: PyTree, key: Key, *, init_samples: Optional[Array] = None):
        return self.sample(params, key, init_samples=init_samples)

    def sample(self, params: PyTree, key: Key, *, init_samples: Optional[Array] = None):
        return _vmc_hmc_sample(
            self.params,
            self.log_prob_fun,
            self.eval_observables,
            params,
            key,
            init_samples,
        )

    def to_dict(self):
        return {k: v for k, v in self.params.__dict__.items() if not callable(v)}


Observable = Callable[[PyTree, Array], Any]


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
    observables: Optional[Sequence[Observable]] = None
) -> VHMC:

    """The variational HMC sampler, allowing for changing variational parameters without
    recompiling the sampler itself. A wrapper around `sampler.sample.HamiltonianMonteCarlo`.

    Parameters
    ----------
    logpsi : Ansatz
        The trial wavefunction. It is expected to either be a callable or have a `.log_prob` method
        which takes variational parameters and a set of samples as arguments, and returns the log-probability.
    n_samples : int
        The number of samples to generate per chain.
    n_chains : int
        The number of independent chains to run.
    warmup : int
        The number of warmup steps to run, optionally adapting
        the momentum metric tensor and the leapfrog step size.
    n_leaps : int
        The number of leapfrog steps to take per sample proposal.
    sweep : int, optional
        The number of samples to discard in between recording a sample, by default 1.
    dims : Sequence[int], optional
        Array dimensions of a single sample, by default None (queried from `logpsi`).
    adapt_step_size : bool, optional
        Whether to adapt the step size during warmup/adaptation, by default True.
    step_size_bounds : Tuple[Scalar, Scalar], optional
        Upper and lower bounds for the leapfrog step size, by default (1e-8, 10.0).
    adapt_metric : bool, optional
        Whether to adapt the momentum metric tensor during warmup/adaptation, by default True.
    diagonal_metric : bool, optional
        Whether to use a diagonal momentum metric tensor (as opposed to a full dense
        covariance matrix), by default True.
    jitter : Scalar, optional
        Jitter to add the leapfrog trajectory length, by default 0.2. Practicaly, that means
        that each time a leapfrog-based proposal is generated, its trajectory length is sampled as:
        length ~ Uniform([(1 - `jitter`) * `n_leaps`, (1 + `jitter`) * `n_leaps`])
    target_acc_rate : Scalar, optional
        The target acceptance rate, by default 0.65. Leapfrog step size is adjusted during
        warmup/adaptation to try to match this rate.
    initial_step_size : Scalar, optional
        The initial leapfrog step size, by default 0.1.
    init_step_size_search : bool, optional
        Whether to perform a fast-and-cheap step size search to find the right order of magnitude
        before adaptation itself, by default True.
    angular : bool, optional
        Whether to treat variables as angles between [-pi, pi], by default False.
    observables : Sequence[Observable], optional
        Observables to evaluate on the samples, by default None. An "observable" is a
        `Callable[[PyTree, Array], Any]` (a callable that takes the variational parameters
        and samples to return a value)

    Returns
    -------
    VHMC
        A callable object that takes variational parameters and a `jax.random.PRNGKey` key and returns
        samples from the encapsulated probability distribution.
    """

    if hasattr(logpsi, "log_prob"):
        log_prob = logpsi.log_prob
    else:
        log_prob = lambda params, x: 2 * jnp.real(logpsi(params, x))

    logp = Partial(log_prob)
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
        chunk_size=None,
    )

    if observables is not None and not isinstance(observables, tuple):

        if hasattr(observables, "__iter__"):
            observables = tuple(observables)
        else:
            observables = (observables,)

    eval_obs = partial(eval_observables, observables)

    return VHMC(logp, hmc_params, eval_obs)


@partial(jax.jit, static_argnames=["hmc_params", "eval_obs"])
def _vmc_hmc_sample(
    hmc_params: HMCParams,
    log_prob_fn: Partial,
    eval_obs: Callable,
    params: PyTree,
    key: Key,
    init_samples: Optional[Array],
):

    logp = Partial(log_prob_fn, params)

    metric = IdentityMetric()
    initial_hamiltonian = Hamiltonian(logp, metric)
    hmc = HMC(hmc_params, initial_hamiltonian)

    samples, info = hmc.sample(key, init_samples)

    observables = eval_obs(params, samples)

    return samples, observables, info
