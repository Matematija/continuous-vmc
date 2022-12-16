from typing import Tuple
from functools import partial

from jax import lax, random
from jax import numpy as jnp

from flax import struct

from .hamiltonian import Hamiltonian
from ..generic import MCMCParams, MCMCState
from ...utils import curry
from ...utils.types import Array, Key, Scalar


@struct.dataclass
class HMCParams(MCMCParams):
    n_leaps: int
    jitter: Scalar


################################################################################


def _leapfrog_body_fn(
    h: Hamiltonian, dt: Scalar, _: int, xp: Tuple[Array, Array]
) -> Tuple[Array, Array]:
    x, p = xp
    x += dt * h.metric(p)
    p -= dt * h.position_grad(x, p)
    return x, p


def leapfrog_proposal(
    h: Hamiltonian, n_leaps: int, dt: Scalar, x: Array, p: Array
) -> Tuple[Array, Array]:

    p -= (dt / 2) * h.position_grad(x, p)

    step_fn = partial(_leapfrog_body_fn, h, dt)
    x, p = lax.fori_loop(0, n_leaps - 1, step_fn, (x, p))

    x += dt * h.metric(p)
    p -= (dt / 2) * h.position_grad(x, p)

    return x, -p


def hmc_mh_accept(
    h: Hamiltonian, x: Array, p: Array, x_: Array, p_: Array, key: Key
) -> Tuple[Scalar, Scalar]:

    log_acc_prob = jnp.minimum(h(x, p) - h(x_, p_), 0.0)
    accepted = log_acc_prob >= jnp.log(random.uniform(key))

    return accepted, jnp.exp(log_acc_prob)


def jitter_trajectory_length(n_leaps: int, jitter: Scalar, key: Key):

    if jitter > 0:
        min_leaps = int((1.0 - jitter) * n_leaps)
        max_leaps = int((1.0 + jitter) * n_leaps)
        return random.randint(key, shape=(), minval=min_leaps, maxval=max_leaps + 1)
    else:
        return n_leaps


@curry
def hmc_kernel(hmc_params: HMCParams, h: Hamiltonian, state: MCMCState, key: Key) -> MCMCState:

    key1, key2, key3 = random.split(key, 3)
    x = state.x

    p = random.normal(key1, shape=x.shape, dtype=x.dtype)
    p = h.metric.transform_normal(p)

    n_leaps = jitter_trajectory_length(hmc_params.n_leaps, hmc_params.jitter, key2)

    x_, p_ = leapfrog_proposal(h, n_leaps, state.step_size, x, p)
    x_, p_ = hmc_params.postprocess_fn(x_, p_)

    accepted, acc_prob = hmc_mh_accept(h, x, p, x_, p_, key3)
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
