from typing import Any, Callable, Optional, Tuple, Union, Sequence
import warnings

from jax import numpy as jnp
from jax.tree_util import tree_leaves

from flax import struct
import optax
from optax.experimental import split_real_and_imaginary

from .vmc import ParameterDerivative
from .hamiltonian import LocalEnergy
from .utils import xpay, tree_size
from .utils.types import Ansatz, PyTree, Key, Array, Scalar

########################################################################################################################


@struct.dataclass
class OptimizerState:

    params: PyTree = struct.field(repr=False)
    grads: Optional[PyTree] = struct.field(repr=False, default=None)
    step: int = 0
    energy: Optional[Scalar] = None
    learning_rate: Optional[Scalar] = None
    observables: Optional[PyTree] = struct.field(repr=False, default=None)
    sampler_info: Optional[PyTree] = struct.field(repr=False, default=None)
    solver_info: Optional[PyTree] = struct.field(repr=False, default=None)
    optax_state: Optional[Any] = struct.field(repr=False, default=None)

    def __repr__(self):

        start = f"{self.__class__.__name__}("
        end = ")"
        middle = ", ".join(
            [
                f"n_params={self.n_params}",
                f"energy={self.energy}",
                f"observables_available={self.observables_available}",
                f"sampler_info_available={self.sampler_info_available}",
                f"solver_info_available={self.solver_info_available}",
            ]
        )

        return start + middle + end

    @property
    def n_params(self):
        return tree_size(self.params)

    @property
    def observables_available(self):
        return self.observables is not None

    @property
    def sampler_info_available(self):
        return self.sampler_info is not None

    @property
    def solver_info_available(self):
        return self.solver_info is not None


def check_optimizer_state(state: OptimizerState, throw: bool = False) -> Sequence[str]:

    """Check if the optimizer state is valid with optional error throwing.

    Parameters
    ----------
    state : OptimizerState
        The optimizer state to check.
    throw : bool, optional
        Whether to throw an error if the state is invalid, by default False.

    Returns
    -------
    Sequence[str]
        A list of error messages if the state is invalid, otherwise an empty list.

    Raises
    ------
    RuntimeError
        If the state is invalid and `throw` is True.
    """

    messages = []

    if any(~jnp.isfinite(leaf).any() for leaf in tree_leaves(state.params)):
        messages.append("Parameters not finite during optimization!")

    if ~jnp.isfinite(state.energy):
        messages.append("Energy not finite during optimization!")

    if not throw:

        for msg in messages:
            warnings.warn(msg)

        return messages
    else:

        if messages:
            raise RuntimeError("\n".join(messages))

        return messages


########################################################################################################################


def StochasticReconfiguration(
    logpsi: Ansatz,
    eloc: LocalEnergy,
    sampler: Callable,
    lr: Union[Scalar, Callable],
    eps: Scalar,
    *,
    solver: str = "shift",
    **solver_kwargs,
) -> Tuple[Callable, Callable]:

    """Stochastic Reconfiguration (SR) optimizer.

    Parameters
    ----------
    logpsi : Ansatz
        The ansatz wavefunction.
    eloc : LocalEnergy
        The local energy object.
    sampler : Callable
        The sampler.
    lr : Union[Scalar, Callable]
        The learning rate / imaginary time step.
    eps : Scalar
        The diagonal-shift regularization parameter, forwarded to the solver.
    solver : str, optional
        The linear solver name as a string. Options:
            * "shift" (default): Apply `eps` as a diagonal shift to the QGT and invert
                using `jax.scipy.linalg.solve` (Cholesky). No special additional `solver_kwargs`
            * "svd": Apply `eps` as a diagonal shift to the QGT and solve the corresponding
                least-squares problem by singular value decomposition (SVD). Additional `solver_kwargs`:
                - `rcond` & `acond` : Singular value (s) cutoff defined as
                    cutoff = maximum(acond, rcond * max(s))
                - `inv_fn` : Alternatively, a function that takes the singular values
                    and returns their regularized inverse
            * "snr": Similarly to "svd", but with a singular value cutoff defined as through the
                "Signal-to-Noise Ratio" method of Schmitt and Heyl (https://arxiv.org/abs/1912.08828).
                Additional `solver_kwargs`:
                - `snr_cutoff` : The Signal-to-Noise Ratio (SNR) cutoff value
                - `exponent` : The exponent of the default SNR pseudoinverse function :
                    `s -> (1 + (cutoff / s) ** (-exponent)`
                - `svd_arcond` & `svd_rcond` : Singular value (s) cutoff defined as
                    cutoff = maximum(acond, rcond * max(s))
                - `reg_fn` : Alternatively, a custom pseudoinverse function that takes the
                    singular values and/or the SNR values and returns the regularized inverse
            * "cg": Solve the linear system using `jax.scipy.sparse.linalg.cg` (Conjugate Gradient).
                Additional `solver_kwargs`: see `jax.scipy.sparse.linalg.cg` documentation.
            * "gmres": Solve the linear system using `jax.scipy.sparse.linalg.gmres` (Generalized Minimal RESidual).
                Additional `solver_kwargs`: see `jax.scipy.sparse.linalg.gmres` documentation.
            * "bicgstab": Solve the linear system using `jax.scipy.sparse.linalg.bicgstab` (Biconjugate Gradient Stabilized).
                Additional `solver_kwargs`: see `jax.scipy.sparse.linalg.bicgstab` documentation.

    Returns
    -------
    Tuple[Callable, Callable]
        The optimizer state initialization function and the optimizer update function.
    """

    params_dot = ParameterDerivative(
        logpsi, eloc, sampler, eps=eps, solver=solver, real_time=False, **solver_kwargs
    )

    init = lambda params: OptimizerState(params=params)

    if callable(lr):
        lr_schedule = lr
    else:
        lr_schedule = optax.constant_schedule(Scalar(lr))

    def kernel(state: OptimizerState, key: Key) -> OptimizerState:

        grads, vmc_info = params_dot(state.params, key, x0=state.grads)
        # Dense solvers will ignore x0

        lr = lr_schedule(state.step)
        params_ = xpay(state.params, grads, -lr)

        return OptimizerState(
            params=params_,
            grads=grads,
            step=state.step + 1,
            energy=vmc_info.energy,
            learning_rate=lr,
            observables=vmc_info.observables,
            sampler_info=vmc_info.sampler_info,
            solver_info=vmc_info.solver_info,
        )

    return init, kernel


QuantumNaturalGradient = StochasticReconfiguration

########################################################################################################################


def _wrap_optax_optimizer(optimizer):
    def wrapped(eloc: LocalEnergy, sampler: Callable, lr: Union[Scalar, Callable], **kwargs):

        if callable(lr):
            lr_schedule = lr
        else:
            lr_schedule = optax.constant_schedule(Scalar(lr))

        optim = split_real_and_imaginary(optimizer(learning_rate=lr_schedule, **kwargs))

        init = lambda params: OptimizerState(params=params, optax_state=optim.init(params))

        def kernel(state: OptimizerState, key: Key) -> Tuple[PyTree, Array]:

            # key, _ = mpi_scatter_keys(key)

            samples, observables, sampler_info = sampler(state.params, key)
            E, grads = eloc.value_and_grad(state.params, samples)

            updates, optax_state = optim.update(grads, state.optax_state)
            new_params = optax.apply_updates(state.params, updates)

            return OptimizerState(
                params=new_params,
                grads=grads,
                step=state.step + 1,
                energy=E,
                learning_rate=lr_schedule(state.step),
                observables=observables,
                sampler_info=sampler_info,
                optax_state=optax_state,
            )

        return init, kernel

    return wrapped


SGD = _wrap_optax_optimizer(optax.sgd)
Adam = _wrap_optax_optimizer(optax.adam)
RMSprop = _wrap_optax_optimizer(optax.rmsprop)
