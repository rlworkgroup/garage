from garage.tf.regressors.base import StochasticRegressor
from garage.tf.regressors.bernoulli_mlp_regressor import (
    BernoulliMLPRegressor)
from garage.tf.regressors.categorical_mlp_regressor import (
    CategoricalMLPRegressor)
from garage.tf.regressors.deterministic_mlp_regressor import (
    DeterministicMLPRegressor)
from garage.tf.regressors.gaussian_conv_regressor import GaussianConvRegressor
from garage.tf.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from garage.tf.regressors.gaussian_mlp_regressor_with_model import (
    GaussianMLPRegressorWithModel)

__all__ = [
    "BernoulliMLPRegressor", "CategoricalMLPRegressor",
    "DeterministicMLPRegressor", "GaussianMLPRegressor",
    "GaussianMLPRegressorWithModel", "GaussianConvRegressor",
    "StochasticRegressor"
]
