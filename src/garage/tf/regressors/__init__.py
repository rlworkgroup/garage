"""Regressors for TensorFlow-based algorithms."""
from garage.tf.regressors.bernoulli_mlp_regressor import BernoulliMLPRegressor
from garage.tf.regressors.categorical_mlp_regressor import (
    CategoricalMLPRegressor)
from garage.tf.regressors.gaussian_cnn_regressor_model import (
    GaussianCNNRegressorModel)
from garage.tf.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from garage.tf.regressors.regressor import Regressor, StochasticRegressor

__all__ = [
    'BernoulliMLPRegressor', 'CategoricalMLPRegressor',
    'GaussianCNNRegressorModel', 'GaussianMLPRegressor', 'Regressor',
    'StochasticRegressor'
]
