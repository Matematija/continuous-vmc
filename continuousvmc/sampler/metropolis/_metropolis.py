# A lot of this file is the same as the HMC code.
# Future plans: extract common code into base classes.

# from typing import Callable, Tuple, Optional
# from functools import partial

# from jax import vmap, jit, tree_map
# from jax import numpy as jnp
# from jax import random, lax
# from jax.tree_util import Partial

# from flax import struct

# from .metric import EuclideanMetric, IdentityMetric, Identity, Metric
# from .optim import WelfordAlgorithm, DualAveraging, DualAveragingState
# from .generic import no_postprocessing, sample_chain, MCMCInfo, MCMCParams

# from ..utils.types import Array, Key, Ansatz, PyTree

# @struct.dataclass
# class RWMState:
#     x: Array
#     accepted: bool
#     acc_prob: float
#     delta: float
#     n_steps: int
#     x_previous: Optional[Array] = None
#     x_last_proposed: Optional[Array] = None

# @struct.dataclass
# class RWMParams:
#     dims: Tuple
#     n_samples: int
#     warmup: int
#     sweep: int
#     adapt_metric: bool
#     target_acc_rate: float = 0.3

# @struct.dataclass
# class RWM:

#     logp: Callable = struct.field(repr=False)
#     params: MCMCParams
#     initial_metric: Metric = struct.field(repr=False)
#     initial_step_size: float = 0.01
#     postprocess_proposal: Callable = struct.field(repr=False, default=no_postprocessing)

#     def __call__(self, inits: Array, key: Key):
#         return rwm_sample(self.params, self.logp, self.initial_metric, self.initial_delta, inits, key, self.postprocess_proposal)

#     def sample(self, inits: Array, key: Key):
#         return rwm_sample(self.params, self.logp, self.initial_metric, self.initial_delta, inits, key, self.postprocess_proposal)

# def RandomWalkMetropolis(
#     log_prob: Callable, dims: Tuple,
#     n_samples: int, warmup: int, sweep: int,
#     target_acc_rate: float = 0.3, initial_delta: float = 0.01,
#     initial_metric: Optional[Metric] = None,
#     adapt_step_size: bool = True, adapt_metric: bool = True,
#     postprocess_proposal: Optional[Callable] = None,
#     **logp_kwargs
#     ) -> RWM:

#     if initial_metric is None:
#         initial_metric = IdentityMetric()

#     logp = Partial(log_prob, **logp_kwargs)
#     rwm_params = MCMCParams(dims, n_samples, warmup, sweep, adapt_step_size, adapt_metric, target_acc_rate)
    
#     if postprocess_proposal is None:
#         postprocess_proposal = no_postprocessing

#     return RWM(logp, rwm_params, initial_metric, initial_step_size, postprocess_proposal)

# def rwm_proposal(metric: Metric, step_size: float, x: Array, key: Key):
#     d = random.normal(key, shape=x.shape, dtype=x.dtype)
#     d = metric.transform_normal(d)
#     return x + step_size*d

# def mh_accept(logp: Callable, x: Array, x_: Array, key: Key):

#     log_acc_prob = jnp.minimum(logp(x_) - logp(x), 0.0)
#     accepted = log_acc_prob >= jnp.log(random.uniform(key))

#     return accepted, jnp.exp(log_acc_prob)

# def rwm_kernel(logp: Callable, metric: Metric, state: RWMState, key: Key, postprocess_fn: Callable = no_postprocessing):

#     key, key_ = random.split(key, 2)
#     x = state.x

#     x_ = rwm_proposal(metric, state.step_size, x, key)
#     x_ = postprocess_fn(x_)

#     accepted, acc_prob = mh_accept(logp, x, x_, key_)

#     x_out = lax.cond(accepted, lambda _: x_, lambda _: x, None)

#     return RWMState(x_out, accepted, acc_prob, state.step_size, state.n_steps+1, x, x_)

# def acc_prob(_: Metric, state: RWMState):
#     return state.acc_prob

# def general_adaptation_window(
#     logp: Callable, metric: Metric, state: RWMState,
#     cost_fun: Callable, target_val: float,
#     optim: DualAveraging, optim_state: DualAveragingState,
#     n_steps: int,
#     key: Key,
#     adapt_metric: bool = True,
#     postprocess_fn: Callable = no_postprocessing,
#     **welford_kwargs
#     ) -> Tuple[RWMState, Metric, Array, Array]:

#     kernel = partial(rwm_kernel, logp, metric, postprocess_fn=postprocess_fn)

#     if adapt_metric:

#         welford = WelfordAlgorithm(**welford_kwargs)
#         welford_state = welford.initialize(*state.x.shape)

#         # if isinstance(metric, Identity):
#         #     metric = metric.to_dense(*state.x.shape)

#     else:
#         welford_state = None

#     init = (state, optim_state, welford_state)

#     def scan_fun(carry, key):

#         state, optim_state, welford_state = carry
#         state = kernel(state, key)

#         cost_val = cost_fun(metric, state)

#         gradient = target_val - cost_val
#         optim_state = optim.update(optim_state, gradient)

#         delta_ = optim_state.x
#         state = state.replace(delta=delta_)

#         if adapt_metric:
#             welford_state = welford.update(welford_state, state.x)

#         return (state, optim_state, welford_state), (delta_, cost_val)

#     key, key_ = random.split(key, 2)
#     init, _ = scan_fun(init, key_)

#     keys = random.split(key, n_steps-1)
#     (out_state, optim_state, welford_state), (deltas, costs) = lax.scan(scan_fun, init, keys)

#     if adapt_metric:
#         metric = EuclideanMetric(welford_state.covariance)

#     return out_state, optim_state, metric, deltas, costs

# def warmup(
#     logp: Callable, metric: Metric, state: RWMState, key: Key,
#     target_acc_rate: float = 0.3, 
#     init_fast: int = 75, init_slow: int = 25, final_fast: int = 50, n_slow_windows: int = 5,
#     postprocess_fn: Callable = no_postprocessing,
#     optim_kwargs: dict = {}, welford_kwargs: dict = {}
#     ):

#     key, key1, key2 = random.split(key, 3)

#     optim = DualAveraging(**optim_kwargs)
#     optim_state = optim.initialize(state.delta)

#     state, optim_state, _, _, _ = general_adaptation_window(
#         logp, metric, state, cost_fun=acc_prob, target_val=target_acc_rate,
#         optim=optim, optim_state=optim_state, n_steps=init_fast, key=key1,
#         adapt_metric=False, postprocess_fn=postprocess_fn, **welford_kwargs
#     )

#     window_sizes = tuple(init_slow*2**n for n in range(n_slow_windows))
#     keys = random.split(key, n_slow_windows)

#     for i, n_steps in enumerate(window_sizes):

#         state, optim_state, metric, _, _ = general_adaptation_window(
#             logp, metric, state, cost_fun=acc_prob, target_val=target_acc_rate,
#             optim=optim, optim_state=optim_state, n_steps=n_steps, key=keys[i],
#             adapt_metric=True, postprocess_fn=postprocess_fn, **welford_kwargs
#         )

#     state, optim_state, _, _, _ = general_adaptation_window(
#         logp, metric, state, cost_fun=acc_prob, target_val=target_acc_rate,
#         optim=optim, optim_state=optim_state, n_steps=final_fast, key=key2,
#         adapt_metric=False, postprocess_fn=postprocess_fn, **welford_kwargs
#     )

#     state = state.replace(delta=optim_state.x_avg)

#     return state, metric

# @partial(vmap, in_axes=(None, None, None, None, 0, 0, None), out_axes=(0, 0))
# def sample_rwm_chain(
#     rwm_params: MCMCParams, logp: Callable, metric: Metric, initial_delta: float, init: Array, key: Key,
#     postprocess_fn: Callable = no_postprocessing
#     ) -> Tuple[Array, MCMCInfo]:

#     key, key_ = random.split(key, 2)
#     state = RWMState(init, True, 1.0, initial_delta, 0)

#     if rwm_params.adapt_metric:
#         window_sizes = make_warmup_schedule(rwm_params.warmup)

#         state, metric = warmup(
#             logp, metric, state, key, rwm_params.target_acc_rate, *window_sizes, postprocess_fn=postprocess_fn
#         )

#         kernel = partial(rwm_kernel, logp, metric, postprocess_fn=postprocess_fn)

#     else:
#         kernel = partial(rwm_kernel, logp, metric, postprocess_fn=postprocess_fn)

#         key, key1 = random.split(key, 2)
#         state = kernel(state, key_)

#         states = sample_chain(kernel, state, key1, n_samples=rwm_params.warmup-1, sweep=1)
#         state = tree_map(lambda l: l[-1], states)

#     samples = sample_chain(kernel, state, key_, n_samples=rwm_params.n_samples, sweep=rwm_params.sweep)

#     if rwm_params.adapt_metric:
#         info = MCMCInfo(samples.accepted.mean(), metric.matrix, 1)
#     else:
#         info = MCMCInfo(samples.accepted.mean(), None, 1)

#     return samples.x, info

# @partial(jit, static_argnames=['rwm_params', 'postprocess_fn'])
# def rwm_sample(
#     rwm_params: MCMCParams, logp: Callable, metric: Metric, initial_delta: float,
#     inits: Array, key: Key, postprocess_fn: Callable = no_postprocessing
#     ) -> Tuple[Array, MCMCInfo]:
    
#     inits = inits.reshape(-1, *rwm_params.dims)
#     n_chains = inits.shape[0]

#     keys = random.split(key, n_chains)
#     samples, info = sample_rwm_chain(rwm_params, logp, metric, initial_delta, inits, keys, postprocess_fn)

#     info = info.replace(n_chains=n_chains)
#     out_samples = samples.reshape(-1, *rwm_params.dims).squeeze()
    
#     return out_samples, info

#################################################################################

# @struct.dataclass
# class VRWM:

#     log_prob_fun: Callable = struct.field(pytree_node=False, repr=False)
#     params: MCMCParams
#     initial_delta: float = 0.01
#     postprocess_proposal: Callable = struct.field(pytree_node=False, repr=False, default=no_postprocessing)

#     def __call__(self, params: PyTree, inits: Array, key: Key):
#         return _vmc_rwm_sample(self.params, self.logp, self.initial_delta, params, inits, key, self.postprocess_proposal)

#     def sample(self, params: PyTree, inits: Array, key: Key):
#         return _vmc_rwm_sample(self.params, self.logp, self.initial_delta, params, inits, key, self.postprocess_proposal)

# def VariationalMetropolis(
#     logpsi: Ansatz,
#     n_samples: int, warmup: int, sweep: int, n_leaps: int,
#     initial_delta: float = 0.01, adapt_metric: bool = True,
#     postprocess_proposal: Optional[Callable] = None
#     ) -> VRWM:

#     rwm_params = MCMCParams(logpsi.dims, n_samples, n_leaps, warmup, sweep, adapt_metric)
    
#     if postprocess_proposal is None:
#         postprocess_proposal = no_postprocessing

#     return VRWM(logpsi.log_prob, rwm_params, initial_delta, postprocess_proposal)

# @partial(jit, static_argnames=['rwm_params', 'postprocess_fn'])
# def _vmc_rwm_sample(
#     rwm_params: MCMCParams, logp: Callable, initial_delta: float,
#     params: PyTree, inits: Array, key: Key,
#     postprocess_fn: Callable = no_postprocessing
#     ):

#     logp = Partial(logp, params)
#     metric = IdentityMetric()

#     rwm = RWM(logp, rwm_params, metric, initial_delta, postprocess_fn)
#     return rwm.sample(inits, key)