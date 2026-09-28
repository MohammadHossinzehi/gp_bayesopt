"""Acquisition functions for minimisation.

Each takes posterior mean ``mu`` and standard deviation ``sigma`` (arrays) and
returns a score where larger means "evaluate here next".
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _pdf(z):
    return _INV_SQRT_2PI * np.exp(-0.5 * z * z)


def expected_improvement(mu, sigma, best, xi=0.01):
    """E[max(best - f - xi, 0)] for f ~ N(mu, sigma^2), in closed form."""
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    imp = best - mu - xi
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, imp / sigma, 0.0)
    ei = imp * ndtr(z) + sigma * _pdf(z)
    return np.where(sigma > 0, np.maximum(ei, 0.0), np.maximum(imp, 0.0))


def probability_of_improvement(mu, sigma, best, xi=0.01):
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, (best - mu - xi) / sigma, 0.0)
    return np.where(sigma > 0, ndtr(z), (best - mu - xi > 0).astype(float))


def lower_confidence_bound(mu, sigma, beta=2.0):
    """Negated LCB so that larger is better, like the other acquisitions."""
    return -(np.asarray(mu, float) - beta * np.asarray(sigma, float))


ACQUISITIONS = {
    "ei": expected_improvement,
    "pi": probability_of_improvement,
    "lcb": lower_confidence_bound,
}
