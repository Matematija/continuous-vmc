from typing import Callable, Tuple

from jax import numpy as jnp
from jax import lax, random
from jax.tree_util import tree_map

from flax import struct

from .optim import DualAveraging, DualAveragingState  # , WelfordAlgorithm
from .metric import Metric, estimate_metric
from .generic import MCMCState, sample_chain
from ..utils.types import Key, Array, Scalar

####################################################################################


@struct.dataclass
class InitialStepSizeState:
    step_size: Scalar
    direction: int
    previous_direction: int
    key: Key


def find_initial_step_size(
    kernel: Callable, hmc_state: MCMCState, key: Key, target_accept: Scalar
) -> Scalar:

    initial_step_size = hmc_state.step_size
    fp_limit = jnp.finfo(lax.dtype(initial_step_size))

    def cond_fun(state: InitialStepSizeState) -> bool:

        not_too_large = (state.step_size < fp_limit.max) | (state.direction <= 0)
        not_too_small = (state.step_size > fp_limit.tiny) | (state.direction >= 0)
        step_size_not_extreme = not_too_large & not_too_small

        not_crossed_threshold = (state.previous_direction == 0) | (
            state.direction == state.previous_direction
        )

        return step_size_not_extreme & not_crossed_threshold

    def body_fun(state: InitialStepSizeState) -> InitialStepSizeState:

        (key,) = random.split(state.key, 1)

        step_size = (2.0**state.direction) * state.step_size

        hmc_state_ = hmc_state.replace(step_size=step_size)
        hmc_state_ = kernel(hmc_state_, key)

        new_direction = lax.cond(
            target_accept < hmc_state_.acc_prob, lambda _: 1, lambda _: -1, None
        )

        return InitialStepSizeState(step_size, new_direction, state.direction, key)

    initial_state = InitialStepSizeState(initial_step_size, 0, 0, key)
    out_state = lax.while_loop(cond_fun, body_fun, initial_state)

    return out_state.step_size


####################################################################################


def stan_warmup_window(
    kernel: Callable,
    state: MCMCState,
    cost_fun: Callable,
    target_val: Scalar,
    n_steps: int,
    key: Key,
    *,
    adapt_metric: bool = True,
    log_step_size_bounds: Tuple[Scalar, Scalar] = (-20.0, 2.0),
    optim_kwargs: dict = {},
    metric_kwargs: dict = {},
) -> Tuple[MCMCState, DualAveragingState, Metric, Array, Array]:

    optim = DualAveraging(**optim_kwargs)
    optim_state = optim.initialize(state.step_size)

    init = (state, optim_state)

    def scan_fun(carry, key):

        hmc_state, optim_state = carry
        hmc_state = kernel(hmc_state, key)

        cost_val = cost_fun(hmc_state)

        gradient = target_val - cost_val
        optim_state = optim.update(optim_state, gradient)

        step_size = jnp.exp(jnp.clip(optim_state.log_x_avg, *log_step_size_bounds))
        hmc_state = hmc_state.replace(step_size=step_size)

        return (hmc_state, optim_state), (hmc_state, step_size, cost_val)

    key, key_ = random.split(key, 2)
    init, _ = scan_fun(init, key_)

    keys = random.split(key, n_steps - 1)
    (out_state, optim_state), (states, dts, costs) = lax.scan(scan_fun, init, keys)

    if adapt_metric:
        metric = estimate_metric(states.x, **metric_kwargs)
    else:
        metric = None

    log_step_size = jnp.clip(optim_state.log_x_avg, *log_step_size_bounds)
    step_size = jnp.exp(log_step_size)

    out_state = out_state.replace(step_size=step_size)

    return out_state, metric, dts, costs


####################################################################################


def metric_adaptation_window(
    kernel: Callable,
    state: MCMCState,
    n_steps: int,
    key: Key,
    *,
    diagonal: bool = False,
    circular: bool = False,
) -> Tuple[MCMCState, Metric]:

    key, key_ = random.split(key, 2)
    state = kernel(state, key_)

    states = sample_chain(kernel, state, key, n_steps - 1)

    out_state = tree_map(lambda l: l[-1], states)
    metric = estimate_metric(states.x, diagonal=diagonal, circular=circular)

    return out_state, metric


def make_stan_warmup_schedule(n_steps: int, n_slow_windows: int = 5):

    init_fast = n_steps // 12
    init_slow = n_steps // 36

    slow_length = init_slow * (2**n_slow_windows - 1)
    assert init_fast + slow_length < n_steps, f"Invalid number of slow windows: {n_slow_windows}"

    final_fast = n_steps - init_fast - slow_length

    return init_fast, init_slow, final_fast, n_slow_windows


def make_warmup_schedule(n_steps: int, n_windows: int = 5):

    init_window = n_steps // (2**n_windows - 1)

    assert init_window > 0, f"Too many windows requested for warmup: {n_windows}"

    windows = tuple(init_window * 2**size for size in range(n_windows - 1))
    last_window = (n_steps - sum(windows),)

    return windows + last_window


####################################################################################


def acc_prob(state: MCMCState):
    return state.acc_prob


def step_size_adaptation(
    kernel: Callable,
    state: MCMCState,
    n_steps: int,
    key: Key,
    target_acc_prob: Scalar,
    *,
    init_step_size_search: bool = True,
    step_size_lims: Tuple[Scalar, Scalar] = (1e-8, 10.0),
    **optim_kwargs,
) -> Tuple[MCMCState, Metric]:

    if init_step_size_search:
        dt = find_initial_step_size(kernel, state, key=key, target_accept=target_acc_prob)
        state = state.replace(step_size=dt)

    out_state, *_ = stan_warmup_window(
        kernel=kernel,
        state=state,
        cost_fun=acc_prob,
        target_val=target_acc_prob,
        n_steps=n_steps,
        key=key,
        adapt_metric=False,
        step_size_lims=step_size_lims,
        optim_kwargs=optim_kwargs,
    )

    return out_state


def vanilla_warmup(kernel: Callable, state: MCMCState, n_steps: int, key: Key):

    key, key_ = random.split(key, 2)
    state = kernel(state, key)

    states = sample_chain(kernel, state, key_, n_steps - 1)
    return tree_map(lambda l: l[-1], states)
