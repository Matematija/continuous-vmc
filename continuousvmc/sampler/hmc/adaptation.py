from typing import Sequence, Tuple

from jax import random

from ..generic import MCMCState
from .kernel import HMCParams, hmc_kernel
from .hamiltonian import Hamiltonian
from ...utils.types import Key, Scalar

from ..warmup import (
    stan_warmup_window,
    metric_adaptation_window,
    step_size_adaptation,
    vanilla_warmup,
    make_stan_warmup_schedule,
    make_warmup_schedule,
    acc_prob,
    find_initial_step_size,
)


def stan_warmup(
    hmc_params: HMCParams,
    h: Hamiltonian,
    state: MCMCState,
    key: Key,
    *,
    target_acc_rate: Scalar = 0.65,
    init_fast: int = 75,
    init_slow: int = 25,
    final_fast: int = 50,
    n_slow_windows: int = 5,
    init_step_size_search: bool = True,
    optim_kwargs: dict = {},
    metric_kwargs: dict = {},
) -> Tuple[MCMCState, Hamiltonian]:

    key, key1, key2 = random.split(key, 3)

    kernel = hmc_kernel(hmc_params, h)

    if init_step_size_search:
        dt = find_initial_step_size(kernel, state, key=key, target_accept=target_acc_rate)
        state = state.replace(step_size=dt)

    state, *_ = stan_warmup_window(
        kernel,
        state,
        cost_fun=acc_prob,
        target_val=target_acc_rate,
        n_steps=init_fast,
        key=key1,
        adapt_metric=False,
        log_step_size_bounds=hmc_params.log_step_size_bounds,
        optim_kwargs=optim_kwargs,
        metric_kwargs=metric_kwargs,
    )

    window_sizes = tuple(init_slow * 2**n for n in range(n_slow_windows))
    keys = random.split(key, n_slow_windows)

    for i, n_steps in enumerate(window_sizes):

        state, metric, *_ = stan_warmup_window(
            kernel,
            state,
            cost_fun=acc_prob,
            target_val=target_acc_rate,
            n_steps=n_steps,
            key=keys[i],
            adapt_metric=True,
            log_step_size_bounds=hmc_params.log_step_size_bounds,
            optim_kwargs=optim_kwargs,
            metric_kwargs=metric_kwargs,
        )

        h = h.replace(metric=metric)
        kernel = hmc_kernel(hmc_params, h)

    state, *_ = stan_warmup_window(
        kernel,
        state,
        cost_fun=acc_prob,
        target_val=target_acc_rate,
        n_steps=final_fast,
        key=key2,
        adapt_metric=False,
        log_step_size_bounds=hmc_params.log_step_size_bounds,
        optim_kwargs=optim_kwargs,
        metric_kwargs=metric_kwargs,
    )

    return state, h


def metric_adaptation(
    hmc_params: HMCParams,
    h: Hamiltonian,
    state: MCMCState,
    windows: Sequence[int],
    key: Key,
    **metric_kwargs,
) -> Tuple[MCMCState, Hamiltonian]:

    for n_steps in windows:

        kernel = hmc_kernel(hmc_params, h)
        (key,) = random.split(key, 1)

        state, metric = metric_adaptation_window(
            kernel=kernel, state=state, n_steps=n_steps, key=key, **metric_kwargs
        )

        h = h.replace(metric=metric)

    h = h.replace(metric=metric)

    return state, h


###############################################################################################


def warmup(
    hmc_params: HMCParams,
    h: Hamiltonian,
    state: MCMCState,
    key: Key,
    *,
    optim_kwargs: dict = {},
    metric_kwargs: dict = {},
) -> Tuple[MCMCState, Hamiltonian]:

    if hmc_params.adapt_metric:

        if hmc_params.adapt_step_size:

            (init_fast, init_slow, final_fast, n_slow_windows) = make_stan_warmup_schedule(
                hmc_params.warmup
            )

            state, h = stan_warmup(
                hmc_params=hmc_params,
                h=h,
                state=state,
                key=key,
                target_acc_rate=hmc_params.target_acc_rate,
                init_fast=init_fast,
                init_slow=init_slow,
                final_fast=final_fast,
                n_slow_windows=n_slow_windows,
                init_step_size_search=hmc_params.init_step_size_search,
                optim_kwargs=optim_kwargs,
                metric_kwargs=metric_kwargs,
            )

        else:
            schedule = make_warmup_schedule(hmc_params.warmup, n_windows=5)
            state, h = metric_adaptation(hmc_params, h, state, schedule, key, **metric_kwargs)

    else:
        kernel = hmc_kernel(hmc_params, h)

        if hmc_params.adapt_step_size:

            state = step_size_adaptation(
                kernel,
                state,
                hmc_params.warmup,
                key,
                hmc_params.target_acc_rate,
                init_step_size_search=hmc_params.init_step_size_search,
                **optim_kwargs,
            )

        else:
            state = vanilla_warmup(kernel, state, hmc_params.warmup, key)

    return state, h
