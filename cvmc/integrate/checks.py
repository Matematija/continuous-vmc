import warnings

from jax import numpy as jnp

from .solver import RungeKuttaIntegrator, IntegratorState
from ..utils.tree import tree_isfinite


def check_integrator_state(
    integrator: RungeKuttaIntegrator, state: IntegratorState, throw: bool = False
):

    messages = []

    if not tree_isfinite(state.y):
        messages.append("The solution is not finite!")

    if ~jnp.isfinite(state.t):
        messages.append(f"Error in incrementing time: time={state.t}")

    if integrator.is_fsal:

        if not tree_isfinite(state.fsal_state.slope):
            messages.append("The FSAL slope is not finite!")

    if integrator.is_adaptive:

        astate = state.adapt_state
        dt_min, dt_max = integrator.dt_bounds

        if astate.dt < dt_min:
            messages.append(f"dt={astate.dt} is smaller than the minimum allowed {dt_min}")

        if astate.dt > dt_max:
            messages.append(f"dt={astate.dt} is larger than the maximum allowed {dt_max}")

        if jnp.isclose(astate.dt, dt_min):
            messages.append(f"dt={astate.dt} is close to the minimum allowed value {dt_min}")

        if jnp.isclose(astate.dt, dt_max):
            messages.append(f"dt={astate.dt} is close to the maximum allowed value {dt_max}")

        if astate.dt < 0:
            messages.append(f"Overflow detected in dt: dt={astate.dt}")

        if not jnp.isfinite(astate.dt):
            messages.append(f"dt={astate.dt} is not finite")

        if astate.n_failed_steps >= integrator.max_failed_steps:
            messages.append(
                f"Maximum number of failed steps {integrator.max_failed_steps} reached: {astate.n_failed_steps}"
            )

    if messages:

        if not throw:

            for msg in messages:
                warnings.warn(msg)

            return messages

        else:
            message = "\n".join(messages)
            raise RuntimeError(message)
