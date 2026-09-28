import numpy as np
import pytest

from gpbo import RBF, ConstantKernel, GaussianProcessRegressor, Matern52, Periodic


def data(n=15, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 5, (n, 1))
    y = np.sin(2 * X[:, 0]) + 0.3 * X[:, 0] + noise * rng.standard_normal(n)
    return X, y


def test_noiseless_interpolation():
    X, y = data()
    gp = GaussianProcessRegressor(RBF(1.0), noise=1e-8, noise_bounds=(1e-10, 1e-6)).fit(X, y)
    mu, sd = gp.predict(X, return_std=True)
    np.testing.assert_allclose(mu, y, atol=1e-4)
    assert sd.max() < 1e-2


def test_lml_matches_dense_formula():
    X, y = data(noise=0.1)
    gp = GaussianProcessRegressor(1.3 * Matern52(0.8), noise=0.05, optimize=False).fit(X, y)
    yn = gp.y_train
    K = gp.kernel(X) + (gp.noise + gp.jitter) * np.eye(len(X))
    sign, logdet = np.linalg.slogdet(K)
    ref = -0.5 * yn @ np.linalg.solve(K, yn) - 0.5 * logdet - 0.5 * len(X) * np.log(2 * np.pi)
    assert sign > 0
    assert gp.log_marginal_likelihood() == pytest.approx(ref, rel=1e-9)


@pytest.mark.parametrize("kernel", [
    ConstantKernel(1.3) * RBF(0.8),
    ConstantKernel(0.7) * Matern52(0.5) + ConstantKernel(0.2) * Periodic(1.0, 2.0),
])
def test_lml_gradient_matches_finite_differences(kernel):
    X, y = data(noise=0.1)
    gp = GaussianProcessRegressor(kernel, noise=0.05, optimize=False).fit(X, y)
    theta = gp.theta
    _, grad = gp.log_marginal_likelihood(theta, eval_gradient=True)
    eps = 1e-6
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = eps
        num = (gp.log_marginal_likelihood(theta + e) - gp.log_marginal_likelihood(theta - e)) / (2 * eps)
        assert grad[i] == pytest.approx(num, rel=1e-4, abs=1e-6)


def test_optimisation_improves_evidence_and_recovers_noise():
    X, y = data(n=60, seed=3, noise=0.2)
    fixed = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(10.0), noise=1e-3, optimize=False).fit(X, y)
    fitted = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(10.0), noise=1e-3, seed=0).fit(X, y)
    assert fitted.log_marginal_likelihood() > fixed.log_marginal_likelihood() + 10
    # noise is learned on the normalised scale; convert back
    noise_sd = np.sqrt(fitted.noise) * fitted._y_std
    assert 0.1 < noise_sd < 0.35


def test_predictive_distribution_is_calibrated():
    X, y = data(n=40, seed=1, noise=0.1)
    gp = GaussianProcessRegressor(ConstantKernel() * Matern52(), noise=1e-2, seed=0).fit(X, y)
    Xt = np.linspace(0.2, 4.8, 200)[:, None]
    truth = np.sin(2 * Xt[:, 0]) + 0.3 * Xt[:, 0]
    mu, sd = gp.predict(Xt, return_std=True)
    assert np.sqrt(np.mean((mu - truth) ** 2)) < 0.1
    z = np.abs(mu - truth) / sd
    assert np.mean(z < 2) > 0.85  # roughly 95% band should contain the latent function


def test_cov_diag_matches_std_and_samples_have_right_moments():
    X, y = data(n=10)
    gp = GaussianProcessRegressor(RBF(1.0), noise=1e-4, optimize=False).fit(X, y)
    Xt = np.linspace(0, 5, 25)[:, None]
    mu, sd = gp.predict(Xt, return_std=True)
    mu2, cov = gp.predict(Xt, return_cov=True)
    np.testing.assert_allclose(mu, mu2)
    np.testing.assert_allclose(np.sqrt(np.clip(np.diag(cov), 0, None)), sd, atol=1e-6)
    S = gp.sample_y(Xt, n_samples=4000, rng=np.random.default_rng(0))
    np.testing.assert_allclose(S.mean(axis=1), mu, atol=0.05)
    np.testing.assert_allclose(S.std(axis=1), sd, atol=0.05)


def test_far_from_data_reverts_to_prior():
    X, y = data(n=10)
    gp = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(0.5), optimize=False).fit(X, y)
    mu, sd = gp.predict(np.array([[100.0]]), return_std=True)
    assert mu[0] == pytest.approx(gp._y_mean, abs=1e-8)
    assert sd[0] == pytest.approx(gp._y_std, rel=1e-6)
