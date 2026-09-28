import numpy as np
import pytest

from gpbo import BayesianOptimizer, expected_improvement, latin_hypercube, random_search
from gpbo.acquisition import lower_confidence_bound, probability_of_improvement
from gpbo.benchmarks import BENCHMARKS


def test_expected_improvement_matches_monte_carlo():
    rng = np.random.default_rng(0)
    mu, sd, best = 0.3, 0.8, 0.5
    f = rng.normal(mu, sd, 2_000_000)
    mc = np.maximum(best - f, 0).mean()
    assert expected_improvement(mu, sd, best, xi=0.0) == pytest.approx(mc, rel=5e-3)


def test_acquisitions_edge_cases():
    assert expected_improvement(1.0, 0.0, best=2.0, xi=0.0) == pytest.approx(1.0)
    assert expected_improvement(3.0, 0.0, best=2.0) == 0.0
    assert probability_of_improvement(0.0, 1.0, best=0.0, xi=0.0) == pytest.approx(0.5)
    # LCB prefers low mean and high uncertainty
    assert lower_confidence_bound(0.0, 1.0) > lower_confidence_bound(0.0, 0.1)
    assert lower_confidence_bound(-1.0, 0.1) > lower_confidence_bound(1.0, 0.1)


def test_latin_hypercube_stratification():
    U = latin_hypercube(10, 3, np.random.default_rng(1))
    for j in range(3):
        assert sorted(np.floor(U[:, j] * 10).astype(int)) == list(range(10))


def test_ask_tell_respects_bounds():
    bounds = [[-2.0, 3.0], [10.0, 11.0]]
    opt = BayesianOptimizer(bounds, seed=0)
    for _ in range(9):
        x = opt.ask()
        assert np.all(x >= [-2, 10]) and np.all(x <= [3, 11])
        opt.tell(x, float(np.sum(x ** 2)))
    assert opt.result().n_evals == 9


def test_unknown_acquisition_raises():
    with pytest.raises(ValueError):
        BayesianOptimizer([[0, 1]], acquisition="nope")


def test_forrester_solved_quickly():
    b = BENCHMARKS["forrester"]
    res = BayesianOptimizer(b.bounds, seed=0).minimize(b.func, 12)
    assert res.fun - b.minimum < 0.05
    assert abs(res.x[0] - 0.7572) < 0.02


@pytest.mark.parametrize("acq", ["ei", "lcb"])
def test_bo_beats_random_search_on_branin(acq):
    b = BENCHMARKS["branin"]
    bo, rs = [], []
    for s in range(3):
        bo.append(BayesianOptimizer(b.bounds, acquisition=acq, seed=s).minimize(b.func, 30).fun)
        rs.append(random_search(b.func, b.bounds, 30, seed=s).fun)
    assert np.median(bo) - b.minimum < 0.3
    assert np.median(bo) < np.median(rs)
