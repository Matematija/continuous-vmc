from typing import Optional, Callable, Sequence, Tuple

import jax
from jax import numpy as jnp
from jax.tree_util import Partial

from flax import struct

from .vmc import ParameterDerivative
from .integrate import RungeKutta, check_integrator_state

from .ansatz import canonicalize_ansatz
from .hamiltonian import LocalEnergy
from .vmc import VMCInfo
from .utils import vmap_chunked, tree_size
from .utils.types import PyTree, Array, Ansatz, Scalar

check_propagator_state = check_integrator_state


@struct.dataclass
class tVMCInfo:
    observables: Optional[Sequence[Array]] = None
    energy: Optional[Scalar] = None
    samples: Optional[Array] = None
    sampler_info: Optional[PyTree] = struct.field(repr=False, default=None)
    solver_info: Optional[PyTree] = struct.field(repr=False, default=None)


def tvmc_callback(y: PyTree, samples: Array, vmc_info: VMCInfo, *__) -> tVMCInfo:
    return vmc_info


#############################################################################

# Maybe move this to the qgt file?
def make_qgt_norm(apply_fn: Callable, chunk_size: Optional[int] = None) -> Callable:

    batched_apply = vmap_chunked(apply_fn, in_axes=(None, 0), chunk_size=chunk_size)

    @Partial
    def qgt_norm(y: PyTree, params: PyTree, samples: PyTree, *_) -> Scalar:

        _, w = jax.jvp(lambda p: batched_apply(p, samples), (params,), (y,))
        w_mean = w.mean(axis=0, keepdims=True)
        # w_mean, token = mpi_allreduce_mean(w_mean)
        wc = w - w_mean

        res2 = jnp.mean(jnp.abs(wc) ** 2)
        # res2, _ = mpi_allreduce_mean(res2, token=token)
        return jnp.sqrt(res2) / tree_size(params)
        # Divide by the number of samples here or not?

    return qgt_norm


def Propagator(
    logpsi: Ansatz,
    eloc: LocalEnergy,
    sampler: Callable,
    eps: Scalar,
    *,
    solver: str = "shift",
    integrator: str = "rk23",
    **kwargs,
) -> Tuple[Callable, Callable]:

    """Construct the time-stepper for the variational wavefunction parameters and its initial state.

    Parameters
    ----------
    logpsi : Ansatz
        The variational wavefunction.
    eloc : LocalEnergy
        The local energy instance.
    sampler : Callable
        The sampler instance.
    eps : Scalar
        The diagonal shift for the S-matrix (Quantum Geometric Tensor).
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
    integrator : str, optional
        The name of the (Runge-Kutta) ODE integrator used for time-stepping. Options:
            * Fixed-step methods: "euler", "midpoint", "heun", "rk4"
            * Adaptive methods: "rk12", "rk12_fehlberg", "rk23" (default), "rk45_fehlberg", "rk45_dopri"
    **kwargs :
        Additional keyword arguments for the solver and the integrator. The keyword arguments of the
        format "solver_{kwarg}" will be passed to the solver and the keyword arguments of the format
        "integrator_{kwarg}" will be passed to the integrator.

    Returns
    -------
    Tuple[Callable, Callable]
        The kernel function and the initializer.
            * Kernel function signature: (state, key) -> state
            * Initializer signature: (initial_params,) -> initial_state

    Raises
    ------
    ValueError
        If any keyword argument is not recognized as starting with "solver_" or "integrator_".
    """

    solver_kwargs = {}
    integrator_kwargs = {}

    for key in kwargs:

        if key.startswith("solver_"):
            key_ = key.replace("solver_", "")
            solver_kwargs[key_] = kwargs[key]
        elif key.startswith("integrator_"):
            key_ = key.replace("integrator_", "")
            integrator_kwargs[key_] = kwargs[key]
        else:
            raise ValueError(f"Unknown keyword argument: {key}")

    params_dot = ParameterDerivative(
        logpsi,
        eloc,
        sampler,
        eps=eps,
        solver=solver,
        real_time=True,
        return_samples=True,
        **solver_kwargs,
    )

    norm_fn = make_qgt_norm(canonicalize_ansatz(logpsi))

    integrator = RungeKutta(
        params_dot,
        name=integrator,
        autonomous=True,
        has_aux=True,
        needs_key=True,
        callback=tvmc_callback,
        norm=norm_fn,
        **integrator_kwargs,
    )

    init = lambda params, t0=0.0, key=None: integrator.initialize(
        params, t0=t0, key=key
    )  # Just for API consistency

    return init, integrator
