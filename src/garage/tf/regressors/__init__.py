"""Regressors for TensorFlow-based algorithms."""
from garage.tf.regressors.base import Regressor
from garage.tf.regressors.base import StochasticRegressor
from garage.tf.regressors.bernoulli_mlp_regressor import BernoulliMLPRegressor
from garage.tf.regressors.categorical_mlp_regressor import (
    CategoricalMLPRegressor)
from garage.tf.regressors.continuous_mlp_regressor import (
    ContinuousMLPRegressor)
from garage.tf.regressors.gaussian_cnn_regressor import GaussianCNNRegressor
from garage.tf.regressors.gaussian_cnn_regressor_model import (
    GaussianCNNRegressorModel)
from garage.tf.regressors.gaussian_mlp_regressor import GaussianMLPRegressor

__all__ = [
    'BernoulliMLPRegressor', 'CategoricalMLPRegressor',
    'ContinuousMLPRegressor', 'GaussianCNNRegressor',
    'GaussianCNNRegressorModel', 'GaussianMLPRegressor', 'Regressor',
    'StochasticRegressor'
]
