import numpy as np
import pytest

from gpbo.kernels import RBF, ConstantKernel, Matern32, Matern52, Periodic

rng = np.random.default_rng(0)
X = rng.random((7, 3))

KERNELS = {
    "rbf": lambda: RBF(0.7),
    "rbf_ard": lambda: RBF([0.3, 1.2, 2.0]),
    "m32": lambda: Matern32(0.5),
    "m52_ard": lambda: Matern52([0.4, 0.9, 1.5]),
    "periodic": lambda: Periodic(0.8, 0.6),
    "scaled": lambda: ConstantKernel(2.5) * Matern52([0.4, 0.9, 1.5]),
    "sum": lambda: RBF(0.5) + 0.3 * Periodic(1.1, 0.4),
    "nested": lambda: 1.7 * (RBF([0.2, 0.5, 1.0]) * Periodic(0.9, 1.3)) + Matern32(0.6),
}


@pytest.mark.parametrize("name", KERNELS)
def test_gradient_matches_finite_differences(name):
    k = KERNELS[name]()
    _, dK = k(X, eval_gradient=True)
    theta, eps = k.theta.copy(), 1e-6
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        k.theta = tp
        Kp = k(X)
        k.theta = tm
        Km = k(X)
        num = (Kp - Km) / (2 * eps)
        np.testing.assert_allclose(dK[:, :, i], num, atol=1e-6, err_msg=k.names[i])
    k.theta = theta


@pytest.mark.parametrize("name", KERNELS)
def test_symmetric_psd_and_diag(name):
    k = KERNELS[name]()
    K = k(X)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-10
    np.testing.assert_allclose(k.diag(X), np.diag(K), atol=1e-12)


@pytest.mark.parametrize("name", KERNELS)
def test_theta_roundtrip_and_bounds_shape(name):
    k = KERNELS[name]()
    t = k.theta
    k.theta = t + 0.1
    np.testing.assert_allclose(k.theta, t + 0.1)
    assert k.bounds.shape == (len(t), 2)
    assert len(k.names) == len(t)


def test_rbf_closed_form():
    a, b = np.array([[0.0, 0.0]]), np.array([[1.0, 2.0]])
    k = RBF([1.0, 2.0])
    assert k(a, b)[0, 0] == pytest.approx(np.exp(-0.5 * (1 + 1)))


def test_matern_limits():
    # Matern kernels equal 1 at zero distance and decay monotonically
    d = np.linspace(0, 5, 50)[:, None]
    for k in (Matern32(1.0), Matern52(1.0)):
        row = k(np.zeros((1, 1)), d)[0]
        assert row[0] == pytest.approx(1.0)
        assert np.all(np.diff(row) < 0)


def test_periodic_is_periodic():
    k = Periodic(0.7, 1.5)
    a = np.array([[0.3]])
    assert k(a, a + 1.5)[0, 0] == pytest.approx(1.0)
    assert k(a, a + 3.0)[0, 0] == pytest.approx(1.0)


def test_cross_covariance_gradient_rejected():
    with pytest.raises(ValueError):
        RBF()(X, X[:2], eval_gradient=True)
