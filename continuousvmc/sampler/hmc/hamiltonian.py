import jax
from jax import numpy as jnp
from jax.tree_util import Partial

from flax import struct

from ..metric import Metric
from ...utils.types import Array


@struct.dataclass
class Hamiltonian:

    logp: Partial = struct.field(repr=False)
    metric: Metric

    def __call__(self, x, p):
        return _eval_hamiltonian(self, x, p)

    def position_grad(self, x, p):
        return _grad_hamiltonian(self, x, p)


@jax.jit
def _eval_hamiltonian(h: Hamiltonian, x: Array, p: Array):
    return 0.5 * jnp.sum(p * h.metric(p)) - h.logp(x)


_grad_hamiltonian = jax.jit(jax.grad(_eval_hamiltonian, argnums=1))
