from typing import Callable, Sequence, Union, Any, Optional

from jax import numpy as jnp

from flax import linen as nn
from .nn.initializers import default_kernel_init, zeros
from .nn.activations import (
    log_i0,
    log_i0_sqrt_pade,
    log_i0_sqrt_taylor,
    log_i0_taylor,
    grad_log_i0_taylor,
)

from .utils.types import Ansatz, Array, Key, PyTree, DType, Scalar
from .utils.types import default_real, is_complex_dtype, real_dtype


class RotorCNN(nn.Module):

    dims: Sequence[int]
    kernels: Union[int, Sequence[int]]
    K: int = 1
    n_features: Optional[int] = None
    use_biases: bool = True
    use_visible_bias: bool = False
    kernel_init: Callable = default_kernel_init(scale=1.0, zero_phase=False)
    bias_init: Callable = zeros
    param_dtype: DType = default_real()

    def setup(self):

        nd = len(self.dims)

        if isinstance(self.kernels, int):
            kernels = [self.kernels]
        else:
            kernels = self.kernels

        self._kernels = [(ks,) * nd if isinstance(ks, int) else ks for ks in kernels]

    @nn.compact
    def __call__(self, thetas: Array):

        thetas = jnp.asarray(thetas)

        k = jnp.arange(1, self.K + 1)
        kthetas = jnp.tensordot(thetas, k, axes=0)

        h = jnp.concatenate([jnp.cos(kthetas), jnp.sin(kthetas)], axis=-1)
        h = jnp.expand_dims(h, axis=0)

        n_features = self.n_features or 2 * self.K

        h = nn.Conv(
            features=n_features,
            kernel_size=self._kernels[0],
            padding="SAME",
            use_bias=self.use_biases,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(h)

        h = log_i0_taylor(h)

        for ks in self._kernels[1:-1]:

            h = nn.Conv(
                features=n_features,
                kernel_size=ks,
                padding="SAME",
                use_bias=self.use_biases,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(h)

            h = grad_log_i0_taylor(h)

        h = nn.Conv(
            features=1,
            kernel_size=self._kernels[-1],
            padding="VALID",
            use_bias=False,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(h)

        out = jnp.sum(h) / jnp.sqrt(h.size)

        if self.use_visible_bias:
            n = jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=0)
            a = self.param("visible_bias", self.bias_init, n.shape, self.param_dtype)
            out += jnp.sum(a * n)

        return out

    def initialize(self, key: Key):
        dtype = real_dtype(self.param_dtype)
        return self.init(key, jnp.zeros(self.dims, dtype=dtype))

    def log_prob(self, params: PyTree, thetas: Array):
        return 2 * jnp.real(self.apply(params, thetas))

    def to_dict(self):
        return dict(
            dims=self.dims,
            kernels=self.kernels,
            K=self.K,
            n_features=self.n_features,
            use_biases=self.use_biases,
            use_visible_bias=self.use_visible_bias,
            param_dtype=self.param_dtype,
        )


class SphericalRBM(nn.Module):

    dims: Sequence[int]
    alpha: Scalar
    activation: str = "original"
    use_visible_bias: bool = True
    use_hidden_bias: bool = True
    kernel_init: Callable = default_kernel_init(scale=0.1)
    bias_init: Callable = zeros
    param_dtype: DType = default_real()

    @nn.compact
    def __call__(self, thetas: Array):

        thetas = jnp.asarray(thetas).ravel()
        n = jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=0)  # n.shape = (nv, 2)

        hidden_features = int(self.alpha * thetas.size)

        x = nn.Dense(
            features=hidden_features,
            use_bias=self.use_hidden_bias,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(n)

        x_norms2 = jnp.sum(x**2, axis=0)

        if self.activation == "original":
            out = log_i0(jnp.sqrt(x_norms2)).sum()
        elif self.activation == "taylor":
            out = log_i0_sqrt_taylor(x_norms2).sum()
        elif self.activation == "pade":
            out = log_i0_sqrt_pade(x_norms2).sum()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        if self.use_visible_bias:

            out += nn.Dense(
                features=1,
                use_bias=False,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
            )(n.reshape(1, -1))

        return jnp.squeeze(out)

    def initialize(self, key: Key):
        dtype = real_dtype(self.param_dtype)
        return self.init(key, jnp.zeros(self.dims, dtype=dtype))

    def log_prob(self, params: PyTree, thetas: Array):
        return 2 * jnp.real(self.apply(params, thetas))

    def to_dict(self):
        return dict(
            dims=self.dims,
            alpha=self.alpha,
            activation=self.activation,
            analytic_activation=self.analytic_activation,
            use_visible_bias=self.use_visible_bias,
            use_hidden_bias=self.use_hidden_bias,
            param_dtype=self.param_dtype,
        )


def canonicalize_kernel_size(kernel_size, dims):

    if kernel_size is not None:

        if isinstance(kernel_size, int):
            return (kernel_size,) * len(dims)
        else:
            return kernel_size

    else:
        return dims


def canonicalize_padding(padding, kernel_size):

    if isinstance(padding, str):

        padding = padding.upper()

        if padding == "FULL":
            padding = tuple(k - 1 for k in kernel_size)

    return padding


class ConvRBM(nn.Module):

    dims: Sequence[int]
    kernel_size: Optional[Union[int, Sequence[int]]] = None
    hidden_features: int = 1
    padding: Union[str, int, Sequence[int]] = "same"
    analytic_activation: bool = False
    use_visible_bias: bool = True
    use_hidden_bias: bool = True
    kernel_init: Callable = default_kernel_init(in_axis=1, out_axis=0)
    bias_init: Callable = zeros
    param_dtype: DType = default_real()
    precision: Any = None

    @nn.compact
    def __call__(self, thetas: Array):

        thetas = jnp.asarray(thetas)
        n = jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=0)  # n.shape = (2, *self.dims)

        kernel_size = canonicalize_kernel_size(self.kernel_size, self.dims)
        padding = canonicalize_padding(self.padding, kernel_size)

        x = nn.Conv(
            features=self.hidden_features,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=False,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )(n[..., None])

        if self.use_hidden_bias:
            hidden_bias_shape = (1, self.hidden_features) + x.shape[2:]
            b = self.param("hidden_bias", self.bias_init, hidden_bias_shape, self.param_dtype)
            x += b

        x_norms2 = jnp.sum(x**2, axis=0)

        if self.activation == "original":
            out = log_i0(jnp.sqrt(x_norms2)).sum()
        elif self.activation == "taylor":
            out = log_i0_sqrt_taylor(x_norms2).sum()
        elif self.activation == "pade":
            out = log_i0_sqrt_pade(x_norms2).sum()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        if self.use_visible_bias:
            a = self.param("visible_bias", self.bias_init, n.shape, self.param_dtype)
            out += jnp.sum(a * n)

        return out

    def initialize(self, key: Key):
        dtype = real_dtype(self.param_dtype)
        return self.init(key, jnp.zeros(self.dims, dtype=dtype))

    def log_prob(self, params: PyTree, thetas: Array):
        return 2 * jnp.real(self.apply(params, thetas))

    def to_dict(self):
        return dict(
            dims=self.dims,
            hidden_features=self.hidden_features,
            padding=self.padding,
            activation=self.activation,
            analytic_activation=self.analytic_activation,
            use_visible_bias=self.use_visible_bias,
            use_hidden_bias=self.use_hidden_bias,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )


class Jastrow(nn.Module):

    dims: Sequence[int]
    use_bias: bool = True
    include_cross: bool = True
    kernel_init: Callable = zeros
    bias_init: Callable = zeros
    param_dtype: DType = default_real()

    @nn.compact
    def __call__(self, thetas: Array):

        thetas = jnp.asarray(thetas).ravel()
        dthetas = thetas[:, None] - thetas[None, :]
        dthetas = dthetas[jnp.triu_indices_from(dthetas, k=1)]
        dot = jnp.cos(dthetas)

        out = nn.Dense(
            features=1,
            use_bias=False,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )(dot)

        if self.include_cross:

            cross = jnp.sin(dthetas)

            out += nn.Dense(
                features=1,
                use_bias=False,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
            )(cross)

        if self.use_bias:

            n = jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=0)

            out += nn.Dense(
                features=1,
                use_bias=False,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
            )(n.ravel())

        return jnp.squeeze(out)

    def initialize(self, key: Key):
        dtype = real_dtype(self.param_dtype)
        return self.init(key, jnp.zeros(self.dims, dtype=dtype))

    def log_prob(self, params: PyTree, thetas: Array):
        return 2 * jnp.real(self.apply(params, thetas))

    @property
    def has_complex_params(self):
        return is_complex_dtype(self.param_dtype)

    def to_dict(self):
        return dict(
            dims=self.dims,
            use_bias=self.use_bias,
            include_cross=self.include_cross,
            initial_params=self.initial_params,
            param_dtype=self.param_dtype,
        )


class Product(nn.Module):

    log_factors: Sequence[Union[Ansatz, Callable]]

    @nn.compact
    def __call__(self, *args, **kwargs):

        out = self.log_factors[0](*args, **kwargs)

        for logpsi in self.log_factors[1:]:
            out += logpsi(*args, **kwargs)

        return out

    def initialize(self, key: Key):
        dtype = real_dtype(self.log_factors[0].param_dtype)
        return self.init(key, jnp.zeros(self.dims, dtype=dtype))

    def log_prob(self, params: PyTree, x: Array):
        return 2 * jnp.real(self.apply(params, x))

    @property
    def dims(self):

        for logpsi in self.log_factors:
            if hasattr(logpsi, "dims"):
                return logpsi.dims

        raise ValueError("`dims` is not defined")

    def to_dict(self):
        return {logpsi.__class__.__name__: logpsi.to_dict() for logpsi in self.log_factors}


def ProductAnsatz(*log_factors):
    return Product(log_factors)


##################################################################


def canonicalize_ansatz(logpsi) -> Callable[[PyTree, Array], Array]:
    return logpsi.apply if hasattr(logpsi, "apply") else logpsi
