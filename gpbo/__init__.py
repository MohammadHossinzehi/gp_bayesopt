"""gpbo: Gaussian process regression and Bayesian optimisation from scratch."""
from .acquisition import expected_improvement, lower_confidence_bound, probability_of_improvement
from .gp import GaussianProcessRegressor
from .kernels import RBF, ConstantKernel, Matern32, Matern52, Periodic, Product, Sum
from .optimizer import BayesianOptimizer, OptimizeResult, latin_hypercube, random_search

__all__ = [
    "GaussianProcessRegressor", "BayesianOptimizer", "OptimizeResult", "random_search",
    "latin_hypercube", "RBF", "Matern32", "Matern52", "Periodic", "ConstantKernel",
    "Sum", "Product", "expected_improvement", "probability_of_improvement",
    "lower_confidence_bound",
]
__version__ = "0.1.0"
