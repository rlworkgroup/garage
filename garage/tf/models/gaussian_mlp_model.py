"""Gaussian MLP Model."""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

from garage.misc import ext
from garage.tf.core.mlp2 import mlp2
from garage.tf.core.parameterLayer import ParameterLayer
from garage.tf.models import PickableModel


class GaussianMLPModel(PickableModel):
    """
    GaussianMLPModel.

    Args:
        :param input_dim: Input dimension.
        :param output_dim: Output dimension.
        :param scope: Variable scope of the model.
        :param hidden_sizes: List of sizes for the fully-connected hidden
          layers.
        :param learn_std: If std is trainable.
        :param init_std: Initial std.
        :param adaptive_std: If std is adaptive (modeled as a mlp).
        :param std_share_network:
        :param std_hidden_sizes: List of sizes for the fully-connected layers
          for std.
        :param min_std: Whether to make sure that the std is at least some
          threshold value, to avoid numerical issues.
        :param std_hidden_nonlinearity: Nonlinearity used for std network.
        :param hidden_nonlinearity: Nonlinearity used for each hidden layer.
        :param output_nonlinearity: Nonlinearity for the output layer.
        :param mean_network: Custom network for the output mean.
        :param std_network: Custom network for the output log std.
        :param std_parametrization: How the std should be parametrized. There
          are a few options:
            - exp: The logarithm of the std will be stored, and applied a
                exponential transformation
            - softplus: The std will be computed as log(1+exp(x))
        :return:
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 dist=tf.contrib.distributions.MultivariateNormalDiag,
                 scope="GaussianMLPModel",
                 hidden_sizes=(32, 32),
                 learn_std=True,
                 init_std=1.0,
                 adaptive_std=False,
                 std_share_network=False,
                 std_hidden_sizes=(32, 32),
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_nonlinearity='relu',
                 hidden_nonlinearity='relu',
                 output_nonlinearity=None,
                 std_parameterization='exp',
                 *args,
                 **kwargs):
        self._scope = scope

        # Network parameters
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_sizes = hidden_sizes
        self._learn_std = learn_std
        self._init_std = init_std
        self._adaptive_std = adaptive_std
        self._std_share_network = std_share_network
        self._std_hidden_sizes = std_hidden_sizes
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._std_parameterization = std_parameterization
        self._dist = dist
        # Tranform std arguments to parameterized space
        self._init_std_param = None
        self._min_std_param = None
        self._max_std_param = None
        if self._std_parameterization == 'exp':
            self._init_std_param = np.log(init_std)
            if min_std:
                self._min_std_param = np.log(min_std)
            if max_std:
                self._max_std_param = np.log(max_std)
        elif self._std_parameterization == 'softplus':
            self._init_std_param = np.log(np.exp(init_std) - 1)
            if min_std:
                self._min_std_param = np.log(np.exp(min_std) - 1)
            if max_std:
                self._max_std_param = np.log(np.exp(max_std) - 1)
        else:
            raise NotImplementedError

        self.model = self.build_model()

    def build_model(self):
        """Build model."""
        input_var = Input(shape=(self._input_dim, ))

        with tf.variable_scope(self._scope):
            if self._std_share_network:
                b = np.concatenate(
                    [
                        np.zeros(self._output_dim),
                        np.full(self._output_dim, self._init_std_param)
                    ],
                    axis=0)
                b = tf.constant_initializer(b)
                mean_std_model = mlp2(
                    input_var=input_var,
                    output_dim=self._output_dim * 2,
                    hidden_sizes=self._hidden_sizes,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    output_nonlinearity=self._output_nonlinearity,
                    output_b_init=b,
                    scope="mean_std_network")
                mean_var = mean_std_model.output[..., self._output_dim:]
                std_param_var = mean_std_model.output[..., :self._output_dim]
            else:
                mean_model = mlp2(
                    input_var=input_var,
                    output_dim=self._output_dim,
                    hidden_sizes=self._hidden_sizes,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    output_nonlinearity=self._output_nonlinearity,
                    scope="mean_network")
                mean_var = mean_model.output

                if self._adaptive_std:
                    std_model = mlp2(
                        input_var=input_var,
                        output_dim=self._output_dim,
                        hidden_sizes=self._std_hidden_sizes,
                        hidden_nonlinearity=self._std_hidden_nonlinearity,
                        output_nonlinearity=self._output_nonlinearity,
                        output_b_init=b,
                        scope="std_network")

                    std_param_var = std_model.output
                else:
                    p = tf.constant_initializer(self._init_std_param)
                    std_param_var = ParameterLayer(
                        length=self._output_dim,
                        initializer=p,
                        trainable=self._learn_std,
                        scope="std_network")(input_var)

            with tf.variable_scope("std_limits"):
                if self._min_std_param:
                    std_param_var = Lambda(
                        lambda x, f, p: f(x, p),
                        arguments={
                            'f': tf.maximum,
                            'p': self._min_std_param
                        })(std_param_var)
                if self._max_std_param:
                    std_param_var = Lambda(
                        lambda x, f, p: f(x, p),
                        arguments={
                            'f': tf.minimum,
                            'p': self._max_std_param
                        })(std_param_var)

        with tf.variable_scope("std_parameterization"):
            if self._std_parameterization == "exp":
                std_var = Lambda(
                    lambda x, f: f(x), arguments={'f': tf.exp})(std_param_var)
            elif self._std_parameterization == "softplus":
                std_var = Lambda(
                    lambda x, f, g: f(1 + g(x)),
                    arguments={
                        'f': tf.log,
                        'g': tf.exp
                    })(std_param_var)
            else:
                raise NotImplementedError

        sample_var = Lambda(
            lambda x, f, s: f(x[0], x[1]).sample(seed=s),
            arguments={
                'f': self._dist,
                's': ext.get_seed()
            })([mean_var, std_param_var])

        return Model(
            inputs=input_var,
            outputs=[mean_var, std_var, std_param_var, sample_var])
