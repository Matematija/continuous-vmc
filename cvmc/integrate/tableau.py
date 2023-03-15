from typing import Tuple, Callable, Optional, Any

from jax import numpy as jnp
from jax.tree_util import tree_map

from flax import struct

from ..utils.types import Array, Key, PyTree, Scalar, default_real
from ..utils.misc import maybe_split


def batched_zeros_like(tree: PyTree, batch_size: int = 1):
    return tree_map(lambda l: jnp.zeros(shape=(batch_size, *l.shape), dtype=l.dtype), tree)


@struct.dataclass
class ButcherTableau:

    order: Tuple[int, int]
    a: Array
    b: Array
    c: Array
    fsal: bool

    @property
    def is_adaptive(self):
        return self.b.ndim == 2


def slopes(
    tableau: ButcherTableau,
    f: Callable,
    init_slope: PyTree,
    init_aux: Any,
    dt: Scalar,
    t: Scalar,
    yt: Array,
    key: Key,
    *args,
):

    times = t + tableau.c * dt

    n_stages = len(tableau.c)
    k = batched_zeros_like(yt, n_stages)
    k = tree_map(lambda leaf, slope: leaf.at[0].set(slope), k, init_slope)

    ki = k
    aux = init_aux

    keys = maybe_split(key, n_stages - 1)

    for i in range(1, n_stages):

        yi = tree_map(lambda yt, k: yt + dt * jnp.tensordot(tableau.a[i], k, axes=1), yt, k)

        ki, *aux = f(times[i], yi, keys[i], *args)
        k = tree_map(lambda k, ki: k.at[i].set(ki), k, ki)

    return k, ki, aux


def step(
    tableau: ButcherTableau,
    f: Callable,
    init_slope: PyTree,
    init_aux: Any,
    dt: Scalar,
    t: Scalar,
    yt: Array,
    key: Optional[Key],
    *args,
):

    k, final_slope, aux = slopes(tableau, f, init_slope, init_aux, dt, t, yt, key, *args)
    b = tableau.b[0] if tableau.b.ndim == 2 else tableau.b

    yp = tree_map(lambda yt_, k: yt_ + dt * jnp.tensordot(b, k, axes=1), yt, k)

    return yp, final_slope, aux


def step_with_error(
    tableau: ButcherTableau,
    f: Callable,
    init_slope: PyTree,
    dt: Scalar,
    t: Scalar,
    yt: Array,
    key: Optional[Key],
    *args,
):

    if not tableau.is_adaptive:
        raise RuntimeError(f"The ODE method is not adaptive!")

    k, final_slope, aux = slopes(tableau, f, init_slope, None, dt, t, yt, key, *args)
    yp = tree_map(lambda y_t, k: y_t + dt * jnp.tensordot(tableau.b[0], k, axes=1), yt, k)

    db = tableau.b[0] - tableau.b[1]
    y_err = tree_map(lambda k: dt * jnp.tensordot(db, k, axes=1), k)

    return yp, y_err, final_slope, aux


####################################################################################

_default_dtype = default_real()

# fmt: off

euler_tableau = ButcherTableau(
    order = (1,),
    a = jnp.zeros((1, 1), dtype=_default_dtype),
    b = jnp.ones((1,), dtype=_default_dtype),
    c = jnp.zeros((1), dtype=_default_dtype),
    fsal = False,
)

midpoint_tableau = ButcherTableau(
    order = (2,),
    a = jnp.array([
            [  0, 0],
            [1/2, 0]
        ],
        dtype=_default_dtype
    ),
    b = jnp.array([0,   1], dtype=_default_dtype),
    c = jnp.array([0, 1/2], dtype=_default_dtype),
    fsal = False,
)

heun_tableau = ButcherTableau(
    order = (2,),
    a = jnp.array([
            [0, 0],
            [1, 0]
        ],
        dtype=_default_dtype
    ),
    b = jnp.array([1/2, 1/2], dtype=_default_dtype),
    c = jnp.array([  0,   1], dtype=_default_dtype),
    fsal = False,
)

rk4_tableau = ButcherTableau(
    order = (4,),
    a = jnp.array([
            [  0,   0, 0, 0],
            [1/2,   0, 0, 0],
            [  0, 1/2, 0, 0],
            [  0,   0, 1, 0]
        ],
        dtype=_default_dtype,
    ),
    b = jnp.array([1/6, 1/3, 1/3, 1/6], dtype=_default_dtype),
    c = jnp.array([0, 1/2, 1/2, 1], dtype=_default_dtype),
    fsal = False,
)

# Adaptive methods:

rk12_tableau = ButcherTableau(
    order = (2, 1),
    a = jnp.array([
            [0, 0],
            [1, 0]
        ],
        dtype=_default_dtype
    ),
    b = jnp.array([
            [1/2, 1/2],
            [  1,   0]
        ],
        dtype=_default_dtype
    ),
    c = jnp.array([0, 1], dtype=_default_dtype),
    fsal = False,
)

rk12_fehlberg_tableau = ButcherTableau(
    order = (2, 1),
    a = jnp.array([
            [    0,       0, 0],
            [  1/2,       0, 0],
            [1/256, 255/256, 0]
        ],
        dtype=_default_dtype
    ),
    b = jnp.array([
            [1/512, 255/256, 1/512],
            [1/256, 255/256,     0]
        ],
        dtype=_default_dtype
    ),
    c = jnp.array([0, 1/2, 2], dtype=_default_dtype),
    fsal = False,
)

# Bogacki–Shampine coefficients
rk23_tableau = ButcherTableau(
    order = (2, 3),
    a = jnp.array(
        [
            [  0,   0,   0, 0],
            [1/2,   0,   0, 0],
            [  0, 3/4,   0, 0],
            [2/9, 1/3, 4/9, 0]
        ],
        dtype=_default_dtype,
    ),
    b = jnp.array([
            [7/24, 1/4, 1/3, 1/8],
            [ 2/9, 1/3, 4/9,   0]
        ],
        dtype=_default_dtype
    ),
    c = jnp.array([0, 1/2, 3/4, 1], dtype=_default_dtype),
    fsal = True,
)

rk45_fehlberg_tableau = ButcherTableau(
    order = (4, 5),
    a = jnp.array(
        [
            [        0,          0,          0,         0,     0, 0],
            [      1/4,          0,          0,         0,     0, 0],
            [     3/32,       9/32,          0,         0,     0, 0],
            [1932/2197, -7200/2197,  7296/2197,         0,     0, 0],
            [  439/216,         -8,   3680/513, -845/4104,     0, 0],
            [    -8/27,          2, -3544/2565, 1859/4104, 11/40, 0],
        ],
        dtype=_default_dtype,
    ),
    b = jnp.array(
        [
            [25/216, 0,  1408/2565,   2197/4104,  -1/5,    0],
            [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55],
        ],
        dtype=_default_dtype,
    ),
    c = jnp.array([0, 1/4, 3/8, 12/13, 1, 1/2], dtype=_default_dtype),
    fsal = False,
)

rk45_dopri_tableau = ButcherTableau(
    order = (5, 4),
    a = jnp.array(
        [
            [         0,           0,          0,        0,           0,     0, 0],
            [       1/5,           0,          0,        0,           0,     0, 0],
            [      3/40,        9/40,          0,        0,           0,     0, 0],
            [     44/45,      -56/15,       32/9,        0,           0,     0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729,           0,     0, 0],
            [ 9017/3168,     -355/33, 46732/5247,   49/176, -5103/18656,     0, 0],
            [    35/384,           0,   500/1113,  125/192,  -2187/6784, 11/84, 0],
        ],
        dtype=_default_dtype,
    ),
    b = jnp.array(
        [
            [    35/384, 0,   500/1113, 125/192,    -2187/6784,    11/84,    0],
            [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40],
        ],
        dtype=_default_dtype,
    ),
    c = jnp.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=_default_dtype),
    fsal = True,
)

# fmt: on

#######################################################################


def get_tableau(name: str) -> ButcherTableau:

    name = name.lower().strip()

    if name == "euler":
        return euler_tableau
    elif name == "midpoint":
        return midpoint_tableau
    elif name == "heun":
        return heun_tableau
    elif name == "rk4":
        return rk4_tableau
    elif name in ["rk12", "heun-euler", "heun euler"]:
        return rk12_tableau
    elif name in ["rk12-fehlberg", "rk12_fehlberg", "rk12 fehlberg"]:
        return rk12_fehlberg_tableau
    elif name in ["rk23", "bogacki-shampine", "bogacki shampine"]:
        return rk23_tableau
    elif name in ["rkf45", "rk45_fehlberg", "fehlberg"]:
        return rk45_fehlberg_tableau
    elif name in ["rk45", "rk45_dopri", "dopri", "dormand-prince", "dormand prince"]:
        return rk45_dopri_tableau
    else:
        raise ValueError(f"Unknown tableau name: {name}")
