"""Covariance functions with analytic gradients.

Every kernel exposes its positive hyperparameters through ``theta``, the
vector of their natural logarithms. Optimising in log space keeps every
parameter positive without constrained optimisation and makes the loss
surface far better conditioned (length scales can span several orders of
magnitude).

``kernel(X, eval_gradient=True)`` returns ``(K, dK)`` where ``dK[:, :, i]``
is the derivative of ``K`` with respect to ``theta[i]``. These gradients feed
the log marginal likelihood gradient in :mod:`gpbo.gp`, and every one of
them is checked against finite differences in ``tests/test_kernels.py``.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "Kernel", "ConstantKernel", "RBF", "Matern32", "Matern52",
    "Periodic", "Sum", "Product",
]


def _as_2d(X):
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


class Kernel:
    """Base class. Subclasses implement ``_params`` / ``_eval``."""

    # names, natural values and natural bounds of hyperparameters
    def _params(self):
        raise NotImplementedError

    def _set_params(self, values):
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return len(self.theta)

    @property
    def theta(self) -> np.ndarray:
        return np.log(np.concatenate([np.atleast_1d(v) for _, v, _ in self._params()]))

    @theta.setter
    def theta(self, theta):
        self._set_params(np.exp(np.asarray(theta, dtype=float)))

    @property
    def bounds(self) -> np.ndarray:
        rows = []
        for _, v, (lo, hi) in self._params():
            rows.extend([(np.log(lo), np.log(hi))] * np.atleast_1d(v).size)
        return np.array(rows, dtype=float)

    @property
    def names(self):
        out = []
        for name, v, _ in self._params():
            size = np.atleast_1d(v).size
            out.extend([name] if size == 1 else [f"{name}[{i}]" for i in range(size)])
        return out

    def __call__(self, X, Y=None, eval_gradient=False):
        X = _as_2d(X)
        if eval_gradient and Y is not None:
            raise ValueError("gradients are only available for K(X, X)")
        Y = X if Y is None else _as_2d(Y)
        return self._eval(X, Y, eval_gradient)

    def diag(self, X):
        X = _as_2d(X)
        return np.array([self._eval(x[None, :], x[None, :], False)[0, 0] for x in X])

    def __add__(self, other):
        return Sum(self, _wrap(other))

    def __radd__(self, other):
        return Sum(_wrap(other), self)

    def __mul__(self, other):
        return Product(self, _wrap(other))

    def __rmul__(self, other):
        return Product(_wrap(other), self)

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def __repr__(self):
        return self._repr()


def _wrap(k):
    return k if isinstance(k, Kernel) else ConstantKernel(float(k))


class ConstantKernel(Kernel):
    """k(x, y) = c. Usually multiplied with a stationary kernel as its signal variance."""

    def __init__(self, value=1.0, bounds=(1e-5, 1e5)):
        self.value = float(value)
        self.value_bounds = bounds

    def _params(self):
        return [("constant", self.value, self.value_bounds)]

    def _set_params(self, v):
        self.value = float(v[0])

    def _eval(self, X, Y, grad):
        K = np.full((X.shape[0], Y.shape[0]), self.value)
        return (K, K[:, :, None].copy()) if grad else K

    def diag(self, X):
        return np.full(_as_2d(X).shape[0], self.value)

    def _repr(self):
        return f"{self.value:.3g}"


class _Stationary(Kernel):
    """Shared machinery for kernels of the scaled distance ``(x - y) / l``.

    ``lengthscale`` may be a scalar (isotropic) or a vector (ARD: automatic
    relevance determination, one scale per input dimension).
    """

    def __init__(self, lengthscale=1.0, bounds=(1e-3, 1e3)):
        self.lengthscale = np.atleast_1d(np.asarray(lengthscale, dtype=float)).copy()
        self.lengthscale_bounds = bounds

    @property
    def ard(self):
        return self.lengthscale.size > 1

    def _params(self):
        return [("lengthscale", self.lengthscale, self.lengthscale_bounds)]

    def _set_params(self, v):
        self.lengthscale = np.asarray(v, dtype=float).copy()

    def _scaled_sq(self, X, Y):
        """Per dimension squared scaled differences, shape (n, m, d)."""
        diff = (X[:, None, :] - Y[None, :, :]) / self.lengthscale
        return diff * diff

    def diag(self, X):
        return np.ones(_as_2d(X).shape[0])

    # f(r2) and g(r2) such that dK/dlog(l_k) = g(r2) * scaled_sq_k
    def _profile(self, r2):
        raise NotImplementedError

    def _eval(self, X, Y, grad):
        sq = self._scaled_sq(X, Y)
        r2 = sq.sum(axis=2)
        K, g = self._profile(r2)
        if not grad:
            return K
        if self.ard:
            dK = g[:, :, None] * sq
        else:
            dK = (g * r2)[:, :, None]
        return K, dK

    def _repr(self):
        ls = np.array2string(self.lengthscale, precision=3) if self.ard else f"{self.lengthscale[0]:.3g}"
        return f"{type(self).__name__}(l={ls})"


class RBF(_Stationary):
    """Squared exponential: k = exp(-r^2 / 2). Infinitely smooth samples."""

    def _profile(self, r2):
        K = np.exp(-0.5 * r2)
        return K, K


class Matern32(_Stationary):
    """Matern nu=3/2: once differentiable samples."""

    def _profile(self, r2):
        r = np.sqrt(np.maximum(r2, 0.0))
        e = np.exp(-np.sqrt(3.0) * r)
        return (1.0 + np.sqrt(3.0) * r) * e, 3.0 * e


class Matern52(_Stationary):
    """Matern nu=5/2: twice differentiable samples. The usual default for BO."""

    def _profile(self, r2):
        r = np.sqrt(np.maximum(r2, 0.0))
        e = np.exp(-np.sqrt(5.0) * r)
        K = (1.0 + np.sqrt(5.0) * r + 5.0 / 3.0 * r2) * e
        g = 5.0 / 3.0 * (1.0 + np.sqrt(5.0) * r) * e
        return K, g


class Periodic(Kernel):
    """Periodic kernel: k = exp(-2 / l^2 * sum_d sin^2(pi (x_d - y_d) / p)).

    The sum over dimensions (MacKay's construction) matters: the common
    shortcut of plugging the Euclidean distance into a single sine is not
    positive semidefinite for inputs with more than one dimension, which the
    PSD test in ``tests/test_kernels.py`` catches.
    """

    def __init__(self, lengthscale=1.0, period=1.0,
                 lengthscale_bounds=(1e-3, 1e3), period_bounds=(1e-3, 1e3)):
        self.lengthscale = float(lengthscale)
        self.period = float(period)
        self.lengthscale_bounds = lengthscale_bounds
        self.period_bounds = period_bounds

    def _params(self):
        return [("lengthscale", self.lengthscale, self.lengthscale_bounds),
                ("period", self.period, self.period_bounds)]

    def _set_params(self, v):
        self.lengthscale, self.period = float(v[0]), float(v[1])

    def diag(self, X):
        return np.ones(_as_2d(X).shape[0])

    def _eval(self, X, Y, grad):
        arg = np.pi * (X[:, None, :] - Y[None, :, :]) / self.period
        s2 = (np.sin(arg) ** 2).sum(axis=2)
        l2 = self.lengthscale ** 2
        K = np.exp(-2.0 * s2 / l2)
        if not grad:
            return K
        d_l = K * 4.0 * s2 / l2
        d_p = K * (2.0 / l2) * (np.sin(2.0 * arg) * arg).sum(axis=2)
        return K, np.stack([d_l, d_p], axis=2)

    def _repr(self):
        return f"Periodic(l={self.lengthscale:.3g}, p={self.period:.3g})"


class _Binary(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1, self.k2 = k1, k2

    @property
    def theta(self):
        return np.concatenate([self.k1.theta, self.k2.theta])

    @theta.setter
    def theta(self, theta):
        n1 = self.k1.n_params
        self.k1.theta = theta[:n1]
        self.k2.theta = theta[n1:]

    @property
    def bounds(self):
        return np.vstack([self.k1.bounds, self.k2.bounds])

    @property
    def names(self):
        return [f"k1.{n}" for n in self.k1.names] + [f"k2.{n}" for n in self.k2.names]


class Sum(_Binary):
    def _eval(self, X, Y, grad):
        if not grad:
            return self.k1._eval(X, Y, False) + self.k2._eval(X, Y, False)
        K1, d1 = self.k1._eval(X, Y, True)
        K2, d2 = self.k2._eval(X, Y, True)
        return K1 + K2, np.concatenate([d1, d2], axis=2)

    def diag(self, X):
        return self.k1.diag(X) + self.k2.diag(X)

    def _repr(self):
        return f"{self.k1!r} + {self.k2!r}"


class Product(_Binary):
    def _eval(self, X, Y, grad):
        if not grad:
            return self.k1._eval(X, Y, False) * self.k2._eval(X, Y, False)
        K1, d1 = self.k1._eval(X, Y, True)
        K2, d2 = self.k2._eval(X, Y, True)
        # product rule
        return K1 * K2, np.concatenate([d1 * K2[:, :, None], d2 * K1[:, :, None]], axis=2)

    def diag(self, X):
        return self.k1.diag(X) * self.k2.diag(X)

    def _repr(self):
        return f"{self.k1!r} * {self.k2!r}"
