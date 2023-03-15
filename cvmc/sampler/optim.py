from jax import numpy as jnp
from flax import struct

from ..utils.types import Scalar


@struct.dataclass
class DualAveragingState:

    log_x: Scalar
    log_x_avg: Scalar
    step: int
    avg_error: Scalar
    mu: Scalar

    @property
    def x(self):
        return jnp.exp(self.log_x)

    @property
    def x_avg(self):
        return jnp.exp(self.log_x_avg)


@struct.dataclass
class DualAveraging:

    t0: Scalar = 10.0
    gamma: Scalar = 0.05
    kappa: Scalar = 0.75

    def initialize(self, x0: Scalar) -> DualAveragingState:
        return DualAveragingState(
            log_x=jnp.log(x0), log_x_avg=0.0, step=1, avg_error=0.0, mu=jnp.log(10 * x0)
        )

    def update(self, state: DualAveragingState, grad: Scalar) -> DualAveragingState:

        reg_step = state.step + self.t0

        eta_t = state.step ** (-self.kappa)
        avg_error = (1 - (1 / reg_step)) * state.avg_error + grad / reg_step

        log_x = state.mu - (jnp.sqrt(state.step) / self.gamma) * avg_error
        log_x_avg = eta_t * state.log_x + (1 - eta_t) * state.log_x_avg

        return DualAveragingState(
            log_x=log_x, log_x_avg=log_x_avg, step=state.step + 1, avg_error=avg_error, mu=state.mu
        )
