from functools import partial
from typing import Optional, Sequence, Union, Any, Tuple

from jax import numpy as jnp
from jax import lax, random
from jax.tree_util import Partial, tree_map

from flax import struct

from .metric import Metric
from ..utils import Array, Key, Scalar, center


def _scan_fun(kernel, state, key):
    state = kernel(state, key)
    return state, state


def _mcmc_sweep(kernel, sweep: int, state, key):

    keys = random.split(key, sweep)
    scan_fun = partial(_scan_fun, kernel)

    state, _ = lax.scan(scan_fun, state, keys)

    return state, state


def sample_chain(kernel, state, key, n_samples: int, sweep: int = 1):

    scan_fn = partial(_mcmc_sweep, kernel, sweep)
    keys = random.split(key, n_samples)

    _, states = lax.scan(scan_fn, state, keys)

    return states


#############################################################################


@Partial
def no_postprocessing(*args):
    return args[0] if len(args) == 1 else args


@Partial
def center_proposal(x: Array, *args):
    return (center(x), *args) if args else center(x)


@Partial
def angular_init_fn(shape: Sequence[int], key: Key, dtype: Any) -> Array:
    return random.uniform(key, shape=shape, minval=-jnp.pi, maxval=jnp.pi).astype(dtype)


@Partial
def randn_init_fn(shape: Sequence[int], key: Key, dtype: Any) -> Array:
    return random.normal(key, shape=shape).astype(dtype)


#############################################################################


@struct.dataclass
class MCMCParams:
    dims: Sequence[int]
    n_samples: int
    n_chains: int
    sweep: int
    warmup: int
    adapt_step_size: bool
    log_step_size_bounds: Tuple[float, float]
    adapt_metric: bool
    init_step_size_search: bool
    target_acc_rate: Scalar
    angular: bool
    diagonal_metric: bool
    initial_step_size: Scalar
    chunk_size: Optional[int]
    init_fn: Optional[Partial] = struct.field(repr=False)
    postprocess_fn: Optional[Partial] = struct.field(repr=False)


@struct.dataclass
class MCMCState:
    x: Array
    accepted: bool
    acc_prob: Scalar
    step_size: Scalar
    n_steps: int
    x_previous: Optional[Array] = None
    x_last_proposed: Optional[Array] = None


@struct.dataclass
class MCMCInfo:

    last_sample: Optional[Array] = None
    acceptances: Optional[Array] = None
    metric: Optional[Metric] = None
    n_chains: Optional[int] = None
    step_size: Optional[Union[Scalar, Array]] = None
    n_steps: Optional[int] = None

    @property
    def mean_acceptance(self):
        return self.acceptances.mean() if self.acceptances is not None else None

    @property
    def mean_step_size(self):
        return self.step_size.mean() if self.step_size is not None else None

    @property
    def mean_metric(self):

        if self.metric is not None:
            return tree_map(partial(jnp.mean, axis=0), self.metric)
        else:
            return None
