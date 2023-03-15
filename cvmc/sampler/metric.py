import numpy as np

from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular

from flax import struct

from .stats import circvar, circcov
from ..utils.types import Array, DType, default_real


@struct.dataclass
class Metric:
    def __matmul__(self, v) -> Array:
        return self(v)

    def transform_normal(self, p: Array) -> Array:
        raise NotImplementedError("Cannot call methods on an abstract class")

    def transform_momentum(self, z: Array) -> Array:
        raise NotImplementedError("Cannot call methods on an abstract class")


@struct.dataclass
class Identity(Metric):
    def transform_normal(self, z: Array) -> Array:
        return z

    def transform_momentum(self, p: Array) -> Array:
        return p

    def to_dense(self, *state_shape, dtype=None):
        n = np.prod(state_shape)
        M = jnp.identity(n, dtype=dtype).reshape(*state_shape, *state_shape)
        return Euclidean(M, M)

    def to_diagonal(self, *state_shape, dtype=None):
        diagonal = jnp.ones(np.prod(state_shape), dtype=dtype)
        return DiagonalEuclidean(diagonal)


def IdentityMetric():
    return Identity()


@struct.dataclass
class Euclidean(Metric):

    matrix: Array = struct.field(repr=False)
    cholesky: Array = struct.field(repr=False)

    def transform_normal(self, z: Array) -> Array:
        return jnp.tensordot(self.matrix, z, axes=self.matrix.ndim // 2)

    def transform_momentum(self, p: Array) -> Array:
        return jnp.tensordot(self.cholesky, p, axes=self.matrix.ndim // 2)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"

    @property
    def shape(self):
        return self.matrix.shape


@struct.dataclass
class DiagonalEuclidean(Metric):

    diagonal: Array = struct.field(repr=False)

    def transform_normal(self, z: Array) -> Array:
        return z * jnp.sqrt(self.diagonal)

    def transform_momentum(self, p: Array) -> Array:
        return p / jnp.sqrt(self.diagonal)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"

    def to_dense(self):
        return Euclidean(jnp.diag(self.diagonal))

    @property
    def shape(self):
        return self.diagonal.shape * 2


def EuclideanMetric(matrix: Array, diagonal: bool = False):

    if not diagonal:

        shape = matrix.shape
        ndim = len(shape) // 2
        flat_shape = np.prod(shape[:ndim])

        cholesky = jnp.linalg.cholesky(matrix.reshape(flat_shape, flat_shape))

        identity = jnp.identity(flat_shape, dtype=matrix.dtype)
        cholesky = solve_triangular(cholesky, identity, lower=True)

        return Euclidean(matrix, cholesky.reshape(*shape))

    else:
        return DiagonalEuclidean(matrix)


################################################################################


def regularize_metric(matrix: Array, sample_size: int, diagonal: bool = False) -> Array:

    reg_cov = (sample_size / (sample_size + 5)) * matrix
    shrinkage = 1e-3 * (5 / (sample_size + 5))

    if diagonal:
        reg_cov += shrinkage

    else:
        s1 = reg_cov.shape[: reg_cov.ndim // 2]
        ndim1 = np.prod(s1)

        ix = jnp.diag_indices(ndim1)
        reg_cov = reg_cov.reshape(ndim1, ndim1).at[ix].add(shrinkage).reshape(*s1, *s1)

    return reg_cov


def estimate_metric(samples: Array, diagonal: bool = False, circular: bool = False) -> Array:

    n_samples, *shape = samples.shape

    assert n_samples > 1, "Must have at least two samples (axis 0) to estimate the covariance"

    flat_samples = samples.reshape(n_samples, -1)

    if not circular:
        if diagonal:
            arr = jnp.var(flat_samples, ddof=1, axis=0).reshape(*shape)
        else:
            arr = jnp.cov(flat_samples, ddof=1, rowvar=False).reshape(*shape, *shape)
    else:
        if diagonal:
            arr = circvar(flat_samples, axis=0).reshape(*shape)
        else:
            arr = circcov(flat_samples).reshape(*shape, *shape)

    arr = regularize_metric(arr, n_samples, diagonal=diagonal)

    return EuclideanMetric(arr, diagonal=diagonal)


################################################################################################
#################### Unused but here for possible future integration ###########################
################################################################################################


@struct.dataclass
class WelfordState:

    mean: Array
    m2: Array
    sample_size: int
    diagonal: bool = False

    @property
    def covariance(self):
        return self.m2 / (self.sample_size - 1)

    @property
    def regularized_covariance(self):
        return regularize_metric(self.covariance, self.sample_size, diagonal=self.diagonal)


class WelfordAlgorithm:

    diagonal: bool = False
    dtype: DType = default_real()

    def initialize(self, *mean_shape):

        mean = jnp.zeros(mean_shape, dtype=self.dtype)

        m2_shape = mean_shape if self.diagonal else mean_shape * 2
        m2 = jnp.zeros(m2_shape, dtype=self.dtype, diagonal=self.diagonal)

        return WelfordState(mean, m2, sample_size=0)

    def update(self, state: WelfordState, value: Array):

        sample_size = state.sample_size + 1

        delta = value - state.mean
        mean = state.mean + delta / sample_size
        updated_delta = value - mean

        if self.diagonal:
            m2 = state.m2 + updated_delta * delta
        else:
            m2 = state.m2 + jnp.tensordot(updated_delta, delta, axes=0)

        return WelfordState(mean, m2, sample_size)
