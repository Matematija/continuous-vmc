from typing import Callable, Sequence, Tuple

from jax import random

from .kernel import rwm_kernel
from ..generic import MCMCParams, MCMCState
from ..metric import Metric
from ...utils.types import Key, Scalar

from ..warmup import (
    find_initial_step_size,
    stan_warmup_window,
    metric_adaptation_window,
    step_size_adaptation,
    vanilla_warmup,
    make_stan_warmup_schedule,
    make_warmup_schedule,
    acc_prob,
)


def stan_warmup(
    logp: Callable,
    rwm_params: MCMCParams,
    metric: Metric,
    state: MCMCState,
    key: Key,
    *,
    target_acc_rate: Scalar = 0.3,
    init_fast: int = 75,
    init_slow: int = 25,
    final_fast: int = 50,
    n_slow_windows: int = 5,
    init_step_size_search: bool = True,
    optim_kwargs: dict = {},
    metric_kwargs: dict = {}
):

    key, key1, key2 = random.split(key, 3)

    kernel = rwm_kernel(rwm_params, logp, metric)

    if init_step_size_search:
        step_size = find_initial_step_size(kernel, state, key=key, target_accept=target_acc_rate)
        state = state.replace(step_size=step_size)

    state, *_ = stan_warmup_window(
        kernel,
        state,
        cost_fun=acc_prob,
        target_val=target_acc_rate,
        n_steps=init_fast,
        key=key1,
        adapt_metric=False,
        log_step_size_bounds=rwm_params.log_step_size_bounds,
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
            log_step_size_bounds=rwm_params.log_step_size_bounds,
            optim_kwargs=optim_kwargs,
            metric_kwargs=metric_kwargs,
        )

        kernel = rwm_kernel(rwm_params, logp, metric)

    state, *_ = stan_warmup_window(
        kernel,
        state,
        cost_fun=acc_prob,
        target_val=target_acc_rate,
        n_steps=final_fast,
        key=key2,
        adapt_metric=False,
        log_step_size_bounds=rwm_params.log_step_size_bounds,
        optim_kwargs=optim_kwargs,
        metric_kwargs=metric_kwargs,
    )

    return state, metric


def metric_adaptation(
    rwm_params: MCMCParams,
    logp: Callable,
    metric: Metric,
    state: MCMCState,
    windows: Sequence[int],
    key: Key,
    **metric_kwargs
) -> Tuple[MCMCState, Metric]:

    keys = random.split(key, len(windows))

    for i, n_steps in enumerate(windows):
        kernel = rwm_kernel(rwm_params, logp, metric)
        state, metric = metric_adaptation_window(kernel, state, n_steps, keys[i], **metric_kwargs)

    return state, metric


###############################################################################################


def warmup(
    rwm_params: MCMCParams,
    logp: Callable,
    metric: Metric,
    state: MCMCState,
    key: Key,
    *,
    optim_kwargs: dict = {},
    metric_kwargs: dict = {}
):

    if rwm_params.adapt_metric:

        if rwm_params.adapt_step_size:

            (init_fast, init_slow, final_fast, n_slow_windows) = make_stan_warmup_schedule(
                rwm_params.warmup
            )

            state, metric = stan_warmup(
                rwm_params=rwm_params,
                logp=logp,
                metric=metric,
                state=state,
                key=key,
                target_acc_rate=rwm_params.target_acc_rate,
                init_fast=init_fast,
                init_slow=init_slow,
                final_fast=final_fast,
                n_slow_windows=n_slow_windows,
                init_step_size_search=rwm_params.init_step_size_search,
                optim_kwargs=optim_kwargs,
                metric_kwargs=metric_kwargs,
            )

        else:
            schedule = make_warmup_schedule(rwm_params.warmup, n_windows=5)
            state, metric = metric_adaptation(
                rwm_params, logp, metric, state, schedule, key, **metric_kwargs
            )

    else:
        kernel = rwm_kernel(rwm_params, logp, metric)

        if rwm_params.adapt_step_size:

            state = step_size_adaptation(
                kernel,
                state,
                rwm_params.warmup,
                key,
                rwm_params.target_acc_rate,
                log_step_size_bounds=rwm_params.log_step_size_bounds,
                **optim_kwargs,
            )

        else:
            state = vanilla_warmup(kernel, state, rwm_params.warmup, key)

    return state, metric
