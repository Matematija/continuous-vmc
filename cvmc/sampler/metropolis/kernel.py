from typing import Callable

from jax import numpy as jnp
from jax import random, lax

from ..metric import Metric
from ..generic import MCMCParams, MCMCState

from ...utils import curry
from ...utils.types import Array, Key, Scalar


def rwm_proposal(metric: Metric, step_size: Scalar, x: Array, key: Key):
    d = random.normal(key, shape=x.shape, dtype=x.dtype)
    d = metric.transform_normal(d)
    return x + step_size * d


def mh_accept(logp: Callable, x: Array, x_: Array, key: Key):

    log_acc_prob = jnp.minimum(logp(x_) - logp(x), 0.0)
    accepted = log_acc_prob >= jnp.log(random.uniform(key))

    return accepted, jnp.exp(log_acc_prob)


@curry
def rwm_kernel(rwm_params: MCMCParams, logp: Callable, metric: Metric, state: MCMCState, key: Key):

    key, key_ = random.split(key, 2)
    x = state.x

    x_ = rwm_proposal(metric, state.step_size, x, key)
    x_ = rwm_params.postprocess_fn(x_)

    accepted, acc_prob = mh_accept(logp, x, x_, key_)

    x_out = lax.cond(accepted, lambda _: x_, lambda _: x, None)

    return MCMCState(
        x_out,
        accepted,
        acc_prob,
        state.step_size,
        state.n_steps + 1,
        x_previous=x,
        x_last_proposed=x_,
    )
