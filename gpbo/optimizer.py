"""Sequential Bayesian optimisation with an ask / tell interface."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .acquisition import ACQUISITIONS
from .gp import GaussianProcessRegressor
from .kernels import ConstantKernel, Matern52


def latin_hypercube(n, d, rng):
    """n points in [0, 1]^d with exactly one point per row/column stratum."""
    cut = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        cut[:, j] = cut[rng.permutation(n), j]
    return cut


@dataclass
class OptimizeResult:
    x: np.ndarray
    fun: float
    X: np.ndarray
    y: np.ndarray
    best_so_far: np.ndarray = field(repr=False)

    @property
    def n_evals(self):
        return len(self.y)


class BayesianOptimizer:
    """Minimise an expensive black box function over a box.

    Inputs are mapped to the unit cube internally so a single set of length
    scale bounds works for every problem, and the default surrogate is a
    Matern 5/2 kernel with one length scale per dimension (ARD).
    """

    def __init__(self, bounds, acquisition="ei", n_initial=None, kernel=None,
                 xi=0.01, beta=2.0, n_candidates=2000, n_polish=5,
                 gp_restarts=2, seed=None):
        self.bounds = np.asarray(bounds, dtype=float).reshape(-1, 2)
        self.dim = self.bounds.shape[0]
        if acquisition not in ACQUISITIONS:
            raise ValueError(f"unknown acquisition {acquisition!r}; pick one of {list(ACQUISITIONS)}")
        self.acquisition = acquisition
        self.n_initial = n_initial if n_initial is not None else 2 * self.dim + 1
        self.kernel = kernel if kernel is not None else ConstantKernel(1.0, (1e-2, 1e2)) * Matern52(np.full(self.dim, 0.3), (1e-2, 1e1))
        self.xi, self.beta = xi, beta
        self.n_candidates, self.n_polish = n_candidates, n_polish
        self.gp_restarts = gp_restarts
        self.rng = np.random.default_rng(seed)
        self._U = []  # unit cube inputs
        self._y = []
        self._initial = list(latin_hypercube(self.n_initial, self.dim, self.rng))
        self.gp = None

    # coordinate transforms
    def _to_unit(self, x):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (np.asarray(x, float) - lo) / (hi - lo)

    def _from_unit(self, u):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + np.asarray(u, float) * (hi - lo)

    def _score(self, U):
        mu, sd = self.gp.predict(np.atleast_2d(U), return_std=True)
        fn = ACQUISITIONS[self.acquisition]
        if self.acquisition == "lcb":
            return fn(mu, sd, beta=self.beta)
        return fn(mu, sd, best=min(self._y), xi=self.xi * max(1.0, np.std(self._y)))

    def ask(self):
        if self._initial:
            return self._from_unit(self._initial.pop())
        self.gp = GaussianProcessRegressor(self.kernel, n_restarts=self.gp_restarts,
                                           seed=int(self.rng.integers(1 << 31)))
        # warm start the next fit from the current hyperparameters
        self.gp.fit(np.array(self._U), np.array(self._y))
        self.kernel = self.gp.kernel.clone()

        cand = self.rng.random((self.n_candidates, self.dim))
        scores = self._score(cand)
        top = cand[np.argsort(scores)[-self.n_polish:]]
        best_u, best_s = top[-1], scores.max()
        box = [(0.0, 1.0)] * self.dim
        for u0 in top:
            res = minimize(lambda u: -self._score(u)[0], u0, method="L-BFGS-B", bounds=box)
            if -res.fun > best_s:
                best_u, best_s = res.x, -res.fun
        # avoid re sampling an existing point exactly (degenerate for the GP)
        if min(np.linalg.norm(np.array(self._U) - best_u, axis=1)) < 1e-6:
            best_u = self.rng.random(self.dim)
        return self._from_unit(np.clip(best_u, 0.0, 1.0))

    def tell(self, x, y):
        self._U.append(self._to_unit(x))
        self._y.append(float(y))

    def minimize(self, func, n_iter, callback=None) -> OptimizeResult:
        for _ in range(n_iter):
            x = self.ask()
            y = float(func(x))
            self.tell(x, y)
            if callback:
                callback(len(self._y), x, y, min(self._y))
        return self.result()

    def result(self) -> OptimizeResult:
        y = np.array(self._y)
        X = self._from_unit(np.array(self._U))
        i = int(np.argmin(y))
        return OptimizeResult(X[i], float(y[i]), X, y, np.minimum.accumulate(y))


def random_search(func, bounds, n_iter, seed=None) -> OptimizeResult:
    """Baseline: uniform random sampling of the box."""
    bounds = np.asarray(bounds, float).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    X = bounds[:, 0] + rng.random((n_iter, len(bounds))) * (bounds[:, 1] - bounds[:, 0])
    y = np.array([float(func(x)) for x in X])
    i = int(np.argmin(y))
    return OptimizeResult(X[i], float(y[i]), X, y, np.minimum.accumulate(y))
