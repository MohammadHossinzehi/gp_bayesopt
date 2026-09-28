"""Exact Gaussian process regression.

The model is ``y = f(x) + eps`` with ``f ~ GP(0, k)`` and ``eps ~ N(0, noise)``.
Everything goes through one Cholesky factorisation of ``K + noise * I``:

* posterior mean   ``k*^T alpha``            where ``alpha = K_y^{-1} y``
* posterior var    ``k** - v^T v``           where ``v = L^{-1} k*``
* log evidence     ``-y^T alpha / 2 - sum(log diag L) - n/2 log(2 pi)``

Hyperparameters (kernel ``theta`` plus ``log noise``) are fitted by
maximising the log marginal likelihood with L-BFGS-B, using its exact
gradient ``1/2 tr((alpha alpha^T - K_y^{-1}) dK/dtheta)`` and a few random
restarts, because the evidence surface is routinely multimodal.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize

from .kernels import ConstantKernel, Kernel, Matern52

LOG_2PI = np.log(2.0 * np.pi)


class GaussianProcessRegressor:
    def __init__(self, kernel: Kernel | None = None, noise: float = 1e-4,
                 noise_bounds=(1e-8, 1.0), optimize: bool = True,
                 n_restarts: int = 3, normalize_y: bool = True,
                 jitter: float = 1e-10, seed=None):
        self.kernel = (kernel if kernel is not None else ConstantKernel(1.0) * Matern52(1.0)).clone()
        self.noise = float(noise)
        self.noise_bounds = noise_bounds
        self.optimize = optimize
        self.n_restarts = n_restarts
        self.normalize_y = normalize_y
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)

    # full hyperparameter vector: [kernel.theta..., log noise]
    @property
    def theta(self):
        return np.append(self.kernel.theta, np.log(self.noise))

    @theta.setter
    def theta(self, t):
        self.kernel.theta = t[:-1]
        self.noise = float(np.exp(t[-1]))

    @property
    def bounds(self):
        return np.vstack([self.kernel.bounds, np.log(self.noise_bounds)])

    # fitting
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.X_train = X[:, None] if X.ndim == 1 else X
        y = np.asarray(y, dtype=float).ravel()
        if self.normalize_y:
            self._y_mean = y.mean()
            self._y_std = y.std() if y.std() > 0 else 1.0
        else:
            self._y_mean, self._y_std = 0.0, 1.0
        self.y_train = (y - self._y_mean) / self._y_std

        if self.optimize and self.kernel.n_params + 1 > 0:
            self._optimize_hyperparameters()
        self._factorize()
        return self

    def _factorize(self):
        K = self.kernel(self.X_train)
        K[np.diag_indices_from(K)] += self.noise + self.jitter
        self.L_ = cholesky(K, lower=True)
        self.alpha_ = cho_solve((self.L_, True), self.y_train)

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """Log evidence of the (normalised) training targets under ``theta``."""
        saved = self.theta
        if theta is not None:
            self.theta = np.asarray(theta, dtype=float)
        try:
            X, y = self.X_train, self.y_train
            n = X.shape[0]
            if eval_gradient:
                K, dK = self.kernel(X, eval_gradient=True)
            else:
                K = self.kernel(X)
            K[np.diag_indices_from(K)] += self.noise + self.jitter
            try:
                L = cholesky(K, lower=True)
            except np.linalg.LinAlgError:
                return (-np.inf, np.zeros(len(saved))) if eval_gradient else -np.inf
            alpha = cho_solve((L, True), y)
            lml = -0.5 * y @ alpha - np.log(np.diag(L)).sum() - 0.5 * n * LOG_2PI
            if not eval_gradient:
                return lml
            # d(noise*I)/d(log noise) = noise * I
            dN = self.noise * np.eye(n)
            dK = np.concatenate([dK, dN[:, :, None]], axis=2)
            inner = np.outer(alpha, alpha) - cho_solve((L, True), np.eye(n))
            grad = 0.5 * np.einsum("ij,jik->k", inner, dK)
            return lml, grad
        finally:
            self.theta = saved

    def _optimize_hyperparameters(self):
        bounds = self.bounds

        def obj(t):
            lml, g = self.log_marginal_likelihood(t, eval_gradient=True)
            if not np.isfinite(lml):
                return 1e25, np.zeros_like(t)
            return -lml, -g

        starts = [np.clip(self.theta, bounds[:, 0], bounds[:, 1])]
        for _ in range(self.n_restarts):
            starts.append(self.rng.uniform(bounds[:, 0], bounds[:, 1]))
        best = None
        for t0 in starts:
            res = minimize(obj, t0, jac=True, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        self.theta = best.x
        self.lml_ = -best.fun

    # prediction
    def predict(self, X, return_std=False, return_cov=False):
        X = np.asarray(X, dtype=float)
        X = X[:, None] if X.ndim == 1 else X
        Ks = self.kernel(X, self.X_train)
        mean = Ks @ self.alpha_ * self._y_std + self._y_mean
        if not (return_std or return_cov):
            return mean
        v = solve_triangular(self.L_, Ks.T, lower=True)
        if return_cov:
            cov = self.kernel(X) - v.T @ v
            return mean, cov * self._y_std ** 2
        var = self.kernel.diag(X) - np.einsum("ij,ij->j", v, v)
        return mean, np.sqrt(np.maximum(var, 0.0)) * self._y_std

    def sample_y(self, X, n_samples=1, rng=None):
        """Joint posterior samples of f at X, shape (len(X), n_samples)."""
        rng = rng or self.rng
        mean, cov = self.predict(X, return_cov=True)
        cov = 0.5 * (cov + cov.T)
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(mean)) * max(1.0, np.abs(np.diag(cov)).max()))
        return mean[:, None] + L @ rng.standard_normal((len(mean), n_samples))
