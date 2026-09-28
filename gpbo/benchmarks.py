"""Standard global optimisation test functions with known minima."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Benchmark:
    name: str
    func: Callable[[np.ndarray], float]
    bounds: np.ndarray
    minimum: float


def forrester(x):
    x = float(np.asarray(x).ravel()[0])
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def branin(x):
    x1, x2 = np.asarray(x, float).ravel()[:2]
    a, b, c = 1.0, 5.1 / (4 * np.pi ** 2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def six_hump_camel(x):
    x1, x2 = np.asarray(x, float).ravel()[:2]
    return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2


_H3_A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
_H3_P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470],
                         [1091, 8732, 5547], [381, 5743, 8828]])
_H_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])


def hartmann3(x):
    x = np.asarray(x, float).ravel()[:3]
    return -float(_H_ALPHA @ np.exp(-(_H3_A * (x - _H3_P) ** 2).sum(axis=1)))


BENCHMARKS = {
    "forrester": Benchmark("forrester", forrester, np.array([[0.0, 1.0]]), -6.020740055767083),
    "branin": Benchmark("branin", branin, np.array([[-5.0, 10.0], [0.0, 15.0]]), 0.39788735772973816),
    "camel": Benchmark("camel", six_hump_camel, np.array([[-3.0, 3.0], [-2.0, 2.0]]), -1.031628453489877),
    "hartmann3": Benchmark("hartmann3", hartmann3, np.array([[0.0, 1.0]] * 3), -3.86278214782076),
}
