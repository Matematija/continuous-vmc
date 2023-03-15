from typing import Callable, Optional, Tuple, Union

import jax
from jax import numpy as jnp
from jax import lax
from jax.tree_util import tree_leaves, Partial

from flax import struct

from .tableau import step, step_with_error, get_tableau, ButcherTableau
from ..utils.tree import tree_size
from ..utils.misc import euclidean_norm, maximum_norm, maybe_split
from ..utils.types import Key, PyTree, Scalar


@struct.dataclass
class FSALState:
    slope: PyTree = struct.field(repr=False)
    info: Optional[PyTree] = None


@struct.dataclass
class AdaptationState:
    dt: Scalar
    accepted: bool = None
    error: Scalar = struct.field(repr=False, default=None)
    y_norm: Scalar = None
    n_failed_steps: Optional[int] = None


@struct.dataclass
class IntegratorState:
    y: PyTree = struct.field(repr=False)
    t: Scalar
    n_rhs_evals: Optional[int] = None
    adapt_state: Optional[AdaptationState] = None
    fsal_state: Optional[FSALState] = None
    info: Optional[PyTree] = None


##########################################################################################


@struct.dataclass
class RungeKuttaIntegrator:

    f: Callable = struct.field(pytree_node=False, repr=False)
    tableau: ButcherTableau = struct.field(repr=False)
    dt: Scalar = 1e-3
    callback: Optional[Callable] = struct.field(pytree_node=False, default=None, repr=False)

    def __call__(self, state: IntegratorState, key: Optional[Key] = None, *args) -> IntegratorState:
        return _fixed_step_integrator(self, state, key, *args)

    def step(self, state: IntegratorState, key: Optional[Key] = None, *args) -> IntegratorState:
        return _fixed_step_integrator(self, state, key, *args)

    def initialize(
        self, init_cond: PyTree, t0: Scalar = 0.0, key: Optional[Key] = None, *args
    ) -> IntegratorState:
        return _init_integrator(self, init_cond, t0, key, *args)

    @property
    def n_stages(self):
        return len(self.tableau.c)

    @property
    def rhs_evals_per_step(self):
        return len(self.tableau.c) - int(self.is_fsal)

    @property
    def is_adaptive(self):
        return False

    @property
    def is_fsal(self):
        return False

    @property
    def error_order(self):
        return None


@struct.dataclass
class AdaptiveRKIntegrator(RungeKuttaIntegrator):

    dt_bounds: Tuple[Optional[Scalar], Optional[Scalar]] = (None, None)
    alpha_bounds: Tuple[Scalar, Scalar] = (0.1, 5.0)
    atol: Scalar = 0.0
    rtol: Scalar = 1e-7
    max_failed_steps: int = 20
    norm_fn: Callable = struct.field(pytree_node=False, default=euclidean_norm, repr=False)

    def __call__(self, state: IntegratorState, key: Optional[Key] = None, *args) -> IntegratorState:
        return _adaptive_step_integrator(self, state, key, *args)

    def step(self, state: IntegratorState, key: Optional[Key] = None, *args) -> IntegratorState:
        return _single_adaptive_step_integrator(self, state, key, *args)

    @property
    def is_adaptive(self):
        return True

    @property
    def error_order(self):
        return self.tableau.order[1]


@struct.dataclass
class AdaptiveRKIntegratorFSAL(AdaptiveRKIntegrator):
    def __call__(self, state: IntegratorState, key: Optional[Key] = None, *args) -> IntegratorState:
        return _adaptive_step_integrator_fsal(self, state, key, *args)

    def step(self, state: IntegratorState, key: Optional[Key] = None, *args) -> IntegratorState:
        return _single_adaptive_step_integrator_fsal(self, state, key, *args)

    @property
    def is_fsal(self):
        return True


def RungeKutta(
    f: Callable,
    name: str = "rk45",
    dt: Scalar = 1e-3,
    dt_bounds: Tuple[Scalar, Scalar] = (1e-8, 1e1),
    alpha_bounds: Tuple[Scalar, Scalar] = (0.1, 5.0),
    atol: Scalar = 0.0,
    rtol: Scalar = 1e-7,
    max_failed_steps: int = 20,
    norm: Union[str, Callable] = "euclidean",
    autonomous: bool = False,
    has_aux: bool = False,
    needs_key: bool = False,
    callback: Optional[Callable] = None,
) -> RungeKuttaIntegrator:

    f_ = canonicalize_rhs_fun(f, autonomous=autonomous, has_aux=has_aux, needs_key=needs_key)
    tableau = get_tableau(name)

    if tableau.is_adaptive:

        norm_fn = _get_norm(norm) if isinstance(norm, str) else Partial(norm)

        if not tableau.fsal:

            return AdaptiveRKIntegrator(
                f_,
                tableau,
                dt,
                callback,
                dt_bounds,
                alpha_bounds,
                atol,
                rtol,
                max_failed_steps,
                norm_fn,
            )

        else:

            return AdaptiveRKIntegratorFSAL(
                f_,
                tableau,
                dt,
                callback,
                dt_bounds,
                alpha_bounds,
                atol,
                rtol,
                max_failed_steps,
                norm_fn,
            )
    else:

        if not tableau.fsal:
            return RungeKuttaIntegrator(f_, tableau, dt, callback)
        else:
            raise RuntimeError("FSAL integrators are not supported for non-adaptive RK methods.")


def _get_norm(name: str):

    if name.strip().lower().startswith("euclid"):
        return lambda y, *_: euclidean_norm(y)
    elif name.strip().lower().startswith("max"):
        return lambda y, *_: maximum_norm(y)
    else:
        raise ValueError(f'Unknown norm "{name}"!')


def canonicalize_rhs_fun(f: Callable, autonomous: bool, has_aux: bool, needs_key: bool) -> Callable:

    if has_aux:
        if needs_key:
            if autonomous:
                canonical_rhs = lambda _, y, key, *args: f(y, key, *args)
            else:
                canonical_rhs = f
        else:
            if autonomous:
                canonical_rhs = lambda _, y, __, *args: f(y, *args)
            else:
                canonical_rhs = lambda t, y, __, *args: f(t, y, *args)
    else:
        if needs_key:
            if autonomous:
                canonical_rhs = lambda _, y, key, *args: (f(y, key, *args), None)
            else:
                canonical_rhs = lambda t, y, key, *args: (f(t, y, key, *args), None)
        else:
            if autonomous:
                canonical_rhs = lambda _, y, __, *args: (f(y, *args), None)
            else:
                canonical_rhs = lambda t, y, __, *args: (f(t, y, *args), None)

    return canonical_rhs


def eval_callback(callback: Optional[Callable], *args, **kwargs):
    return callback(*args, **kwargs) if callback is not None else None


########################################################################################################################


def _init_integrator(
    integrator: RungeKuttaIntegrator, y0: PyTree, t0: Scalar = 0.0, key: Optional[Key] = None, *args
) -> IntegratorState:

    if not all(jnp.all(jnp.isfinite(l)) for l in tree_leaves(y0)):
        raise RuntimeError("Initial condition to the ODE solver must be finite!")

    if integrator.is_fsal:
        first_slope, *aux = integrator.f(t0, y0, key, *args)
        info = eval_callback(integrator.callback, y0, *aux)
        fsal_state = FSALState(first_slope, info)
        n_rhs_evals = 1
    else:
        info = None
        fsal_state = None
        n_rhs_evals = 0

    if integrator.is_adaptive:  # has to be fsal to be adaptive

        if integrator.is_fsal:
            y_norm = integrator.norm_fn(y0, y0, *aux)
        else:
            try:
                y_norm = integrator.norm_fn(y0)
            except TypeError:
                y_norm = euclidean_norm(y0)

        adapt_state = AdaptationState(
            dt=integrator.dt, accepted=True, y_norm=y_norm, n_failed_steps=0
        )

    else:
        adapt_state = None

    return IntegratorState(
        y0, t0, n_rhs_evals=n_rhs_evals, fsal_state=fsal_state, adapt_state=adapt_state
    )


#######################################################################################################
##################################### Fixed step integrators: #########################################
#######################################################################################################


@jax.jit
def _fixed_step_integrator(
    integrator: RungeKuttaIntegrator, state: IntegratorState, *args, key: Optional[Key]
) -> IntegratorState:

    key1, key2 = maybe_split(key, 2)

    first_slope, *aux = integrator.f(state.t, state.y, key1, *args)
    info = eval_callback(integrator.callback, state.y, *aux)

    y, _, _ = step(
        integrator.tableau,
        integrator.f,
        first_slope,
        aux,
        integrator.dt,
        state.t,
        state.y,
        key2,
        *args,
    )

    n_rhs_evals = state.n_rhs_evals + integrator.rhs_evals_per_step

    return IntegratorState(y=y, t=state.t + integrator.dt, n_rhs_evals=n_rhs_evals, info=info)


#######################################################################################################
###################################### Adaptive integrators: ##########################################
#######################################################################################################


def _scale_error(yp, y_err, atol, rtol, norm_fn, last_norm_y, *aux):

    norm_y = norm_fn(yp, yp, *aux)
    scale = (atol + jnp.maximum(norm_y, last_norm_y) * rtol) / tree_size(y_err)

    return norm_fn(y_err, yp, *aux) / scale, norm_y


def _propose_dt(
    dt: Scalar, scaled_error: Scalar, error_order: int, dt_limits: Tuple[Scalar, Scalar]
) -> Scalar:

    safety_factor = 0.95
    err_exponent = -1.0 / (1 + error_order)

    return jnp.clip(dt * safety_factor * scaled_error**err_exponent, *dt_limits)


@jax.jit
def _single_adaptive_step_integrator(
    integrator: AdaptiveRKIntegrator, state: IntegratorState, key: Optional[Key] = None, *args
) -> IntegratorState:

    key1, key2 = maybe_split(key, 2)

    first_slope, *aux = integrator.f(state.t, state.y, key1, *args)
    info = eval_callback(integrator.callback, state.y, *aux)

    dt_min, dt_max = integrator.dt_bounds
    alpha_min, alpha_max = integrator.alpha_bounds

    dtp = state.adapt_state.dt  # proposed dt
    dt = jnp.minimum(dtp, dt_max) if dt_max is not None else dtp

    yp, err, _, last_aux = step_with_error(
        integrator.tableau, integrator.f, first_slope, dt, state.t, state.y, key2, *args
    )

    scaled_err, y_norm = _scale_error(
        yp,
        err,
        integrator.atol,
        integrator.rtol,
        integrator.norm_fn,
        state.adapt_state.y_norm,
        *last_aux,
    )

    adjusted_dt_limits = (
        jnp.maximum(alpha_min * dtp, dt_min) if dt_min is not None else alpha_min * dtp,
        jnp.minimum(alpha_max * dtp, dt_max) if dt_max is not None else alpha_max * dtp,
    )

    dtp = _propose_dt(dt, scaled_err, integrator.error_order, adjusted_dt_limits)  # new proposed dt

    min_dt_reached = jnp.isclose(dtp, dt_min) if dt_min is not None else False
    accept = jnp.logical_or(scaled_err < 1.0, min_dt_reached)

    # accept, _ = mpi_allreduce_all(accept)

    y = lax.cond(accept, lambda _: yp, lambda _: state.y, None)
    t = lax.cond(accept, lambda _: state.t + dt, lambda _: state.t, None)

    n_failed_steps = lax.cond(
        accept, lambda _: 0, lambda i: i + 1, state.adapt_state.n_failed_steps
    )

    adapt_state = AdaptationState(
        dt=dtp, accepted=accept, error=err, y_norm=y_norm, n_failed_steps=n_failed_steps
    )

    n_rhs_evals = state.n_rhs_evals + integrator.rhs_evals_per_step

    return IntegratorState(y=y, t=t, n_rhs_evals=n_rhs_evals, adapt_state=adapt_state, info=info)


@jax.jit
def _single_adaptive_step_integrator_fsal(
    integrator: AdaptiveRKIntegrator, state: IntegratorState, key: Optional[Key] = None, *args
) -> IntegratorState:

    first_slope, info = state.fsal_state.slope, state.fsal_state.info

    dt_min, dt_max = integrator.dt_bounds
    alpha_min, alpha_max = integrator.alpha_bounds

    dtp = state.adapt_state.dt  # proposed dt
    dt = jnp.minimum(dtp, dt_max) if dt_max is not None else dtp

    yp, err, last_slope, aux = step_with_error(
        integrator.tableau, integrator.f, first_slope, dt, state.t, state.y, key, *args
    )

    scaled_err, y_norm = _scale_error(
        yp,
        err,
        integrator.atol,
        integrator.rtol,
        integrator.norm_fn,
        state.adapt_state.y_norm,
        *aux,
    )

    adjusted_dt_limits = (
        jnp.maximum(alpha_min * dtp, dt_min) if dt_min is not None else alpha_min * dtp,
        jnp.minimum(alpha_max * dtp, dt_max) if dt_max is not None else alpha_max * dtp,
    )

    dtp = _propose_dt(dt, scaled_err, integrator.error_order, adjusted_dt_limits)  # new proposed dt

    min_dt_reached = jnp.isclose(dtp, dt_min) if dt_min is not None else False
    accept = jnp.logical_or(scaled_err < 1.0, min_dt_reached)

    # accept, _ = mpi_allreduce_all(accept)

    y = lax.cond(accept, lambda _: yp, lambda _: state.y, None)
    t = lax.cond(accept, lambda _: state.t + dt, lambda _: state.t, None)

    n_failed_steps = lax.cond(
        accept, lambda _: 0, lambda i: i + 1, state.adapt_state.n_failed_steps
    )

    n_rhs_evals = state.n_rhs_evals + integrator.rhs_evals_per_step

    def update_fsal_state(_):
        last_info = eval_callback(integrator.callback, yp, *aux)
        return FSALState(last_slope, last_info)

    fsal_state = lax.cond(accept, update_fsal_state, lambda _: state.fsal_state, None)

    adapt_state = AdaptationState(
        dt=dtp, accepted=accept, error=err, y_norm=y_norm, n_failed_steps=n_failed_steps
    )

    return IntegratorState(
        y=y, t=t, n_rhs_evals=n_rhs_evals, adapt_state=adapt_state, fsal_state=fsal_state, info=info
    )


def step_until_accept(step_fn: Callable):
    @jax.jit
    def wrapped(
        integrator: AdaptiveRKIntegrator, state: IntegratorState, key: Optional[Key], *args
    ) -> IntegratorState:
        def cond_fun(arg):
            state, _ = arg
            return jnp.logical_and(
                ~state.adapt_state.accepted,
                state.adapt_state.n_failed_steps < integrator.max_failed_steps,
            )

        def body_fun(arg):
            state, key = arg
            state_ = step_fn(integrator, state, key, *args)
            (key_,) = maybe_split(key, 1)
            return (state_, key_)

        key, key_ = maybe_split(key, 2)
        init_state = step_fn(integrator, state, key, *args)

        state, _ = lax.while_loop(cond_fun, body_fun, (init_state, key_))

        return state

    return wrapped


_adaptive_step_integrator = step_until_accept(_single_adaptive_step_integrator)
_adaptive_step_integrator_fsal = step_until_accept(_single_adaptive_step_integrator_fsal)
