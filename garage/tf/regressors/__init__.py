from garage.tf.regressors.base import StochasticRegressor
from garage.tf.regressors.base2 import Regressor2
from garage.tf.regressors.base2 import StochasticRegressor2
from garage.tf.regressors.bernoulli_mlp_regressor import (
    BernoulliMLPRegressor)
from garage.tf.regressors.categorical_mlp_regressor import (
    CategoricalMLPRegressor)
from garage.tf.regressors.continuous_mlp_regressor import (
    ContinuousMLPRegressor)
from garage.tf.regressors.continuous_mlp_regressor_with_model import (
    ContinuousMLPRegressorWithModel)
from garage.tf.regressors.gaussian_conv_regressor import GaussianConvRegressor
from garage.tf.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from garage.tf.regressors.gaussian_mlp_regressor_model import (
    GaussianMLPRegressorModel)
from garage.tf.regressors.gaussian_mlp_regressor_with_model import (
    GaussianMLPRegressorWithModel)

__all__ = [
    'BernoulliMLPRegressor', 'CategoricalMLPRegressor',
    'ContinuousMLPRegressor', 'ContinuousMLPRegressorWithModel',
    'GaussianMLPRegressor', 'GaussianMLPRegressorModel',
    'GaussianMLPRegressorWithModel', 'GaussianConvRegressor', 'Regressor2',
    'StochasticRegressor', 'StochasticRegressor2'
]
