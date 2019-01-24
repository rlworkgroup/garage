import numpy as np
import tensorflow as tf

from garage.tf.core.mlp2 import mlp2
from garage.tf.core.parameterLayer import ParameterLayer
from garage.tf.core.distributionLayer import DistributionLayer
from garage.tf.models import Model as GarageModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda

# flake8: noqa
# pylint: noqa


class GaussianMLPModel2(GarageModel):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dist=tf.contrib.distributions.MultivariateNormalDiag,
                 name="GaussianMLPModel",
                 hidden_sizes=(32, 32),
                 learn_std=True,
                 init_std=1.0,
                 adaptive_std=False,
                 std_share_network=False,
                 std_hidden_sizes=(32, 32),
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_nonlinearity=tf.nn.tanh,
                 hidden_nonlinearity=tf.nn.tanh,
                 output_nonlinearity=None,
                 std_parameterization='exp',
                 *args,
                 **kwargs):
        """
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param name: name of the model
        :param hidden_sizes: list of sizes for the fully-connected hidden
          layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers
          for std
        :param min_std: whether to make sure that the std is at least some
          threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parametrization: how the std should be parametrized. There
          are a few options:
            - exp: the logarithm of the std will be stored, and applied a
                exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :return:
        """
        self.name = name
        # self._variable_scope = tf.variable_scope(
        #     self.name, reuse=tf.AUTO_REUSE)
        # self._name_scope = tf.name_scope(self.name)

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
        self._dist = tf.contrib.distributions.MultivariateNormalDiag
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
        self._mean = self.model.outputs[0]
        self._std = self.model.outputs[1]
        self._std_param = self.model.outputs[2]
        self._sample_var = self.model.outputs[3]

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def std_param(self):
        return self._std_param

    @property
    def dist(self):
        return self._dist

    @property
    def sample(self):
        return self._sample_var

    def build_model(self):
        input_var = Input(shape=(self._input_dim, ))

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
                name="mean_std_network")
            mean_var = mlp2.output[..., self._output_dim:]
            std_param_var = mlp2.output[..., :self._output_dim]
        else:
            mean_model = mlp2(
                input_var=input_var,
                output_dim=self._output_dim,
                hidden_sizes=self._hidden_sizes,
                hidden_nonlinearity=self._hidden_nonlinearity,
                output_nonlinearity=self._output_nonlinearity,
                name="mean_network")
            mean_var = mean_model.output

            if self._adaptive_std:
                b = tf.constant_initializer(self._init_std_param)
                std_model = mlp2(
                    input_var=input_var,
                    output_dim=self._output_dim,
                    hidden_sizes=self._std_hidden_sizes,
                    hidden_nonlinearity=self._std_hidden_nonlinearity,
                    output_nonlinearity=self._output_nonlinearity,
                    output_b_init=b,
                    name="std_network")

                std_param_var = std_model.output
            else:
                p = tf.constant_initializer(self._init_std_param)
                std_param_var = ParameterLayer(
                    length=self._output_dim,
                    initializer=p,
                    trainable=self._learn_std,
                    name="std_network")(input_var)

        with tf.variable_scope("std_limits"):
            if self._min_std_param:
                std_param_var = Lambda(lambda x: tf.maximum(
                    x, self._min_std_param))(std_param_var)
            if self._max_std_param:
                std_param_var = Lambda(lambda x: tf.minimum(
                    x, self._max_std_param))(std_param_var)

        with tf.variable_scope("std_parameterization"):
            if self._std_parameterization == "exp":
                std_var = Lambda(lambda x: tf.exp(x))(std_param_var)
            elif self._std_parameterization == "softplus":
                std_var = Lambda(lambda x: tf.log(1. + tf.exp(x)))(
                    std_param_var)
            else:
                raise NotImplementedError

        sample_var = DistributionLayer(self._dist)([mean_var, std_var])

        return Model(
            inputs=input_var,
            outputs=[mean_var, std_var, std_param_var, sample_var])
