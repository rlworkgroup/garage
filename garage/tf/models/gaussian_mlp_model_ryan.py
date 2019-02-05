import collections

import numpy as np
import tensorflow as tf

from garage.misc import ext
from garage.tf.core.mlp import mlp
from garage.tf.core.parameter import parameter
from garage.tf.models.base import TfModel

class GaussianMLPModel(TfModel):
    def __init__(self,
                 name=None,
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
                 std_parameterization='exp'):
        """
        :param env_spec:
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
        super().__init__(name)


        # Network parameters
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

    @property
    def init_spec(self):
        return (), {
            'name': self._name,
            'hidden_sizes': self._hidden_sizes,
            'learn_std': self._learn_std,
            'init_std': self._init_std,
            'adaptive_std': self._adaptive_std,
            'std_share_network': self._std_share_network,
            'std_hidden_sizes': self._std_hidden_sizes,
            'min_std': self._min_std,
            'max_std': self._max_std,
            'std_hidden_nonlinearity': self._std_hidden_nonlinearity,
            'hidden_nonlinearity': self._hidden_nonlinearity,
            'output_nonlinearity': self._output_nonlinearity,
            'std_parameterization': self._std_parameterization,
        }

    def _build(self, inputs):
        action_dim = 4
        small = 1e-5

        state_input = inputs[0]

        with tf.variable_scope("dist_params"):
            if self._std_share_network:
                # mean and std networks share an MLP
                b = np.concatenate(
                    [
                        np.zeros(action_dim),
                        np.full(action_dim, self._init_std_param)
                    ],
                    axis=0)
                b = tf.constant_initializer(b)
                mean_std_network = mlp(
                    state_input,
                    output_dim=action_dim * 2,
                    hidden_sizes=self._hidden_sizes,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    output_nonlinearity=self._output_nonlinearity,
                    # hidden_w_init=tf.orthogonal_initializer(1.0),
                    # output_w_init=tf.orthogonal_initializer(0.1),
                    output_b_init=b,
                    name="mean_std_network")
                with tf.variable_scope("mean_network"):
                    mean_network = mean_std_network[..., :action_dim]
                with tf.variable_scope("std_network"):
                    std_network = mean_std_network[..., action_dim:]

            else:
                # separate MLPs for mean and std networks
                # mean network
                mean_network = mlp(
                    state_input,
                    output_dim=action_dim,
                    hidden_sizes=self._hidden_sizes,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    output_nonlinearity=self._output_nonlinearity,
                    name="mean_network")

                # std network
                if self._adaptive_std:
                    b = tf.constant_initializer(self._init_std_param)
                    std_network = mlp(
                        state_input,
                        output_dim=action_dim,
                        hidden_sizes=self._std_hidden_sizes,
                        hidden_nonlinearity=self._std_hidden_nonlinearity,
                        output_nonlinearity=self._output_nonlinearity,
                        output_b_init=b,
                        name="std_network")
                else:
                    p = tf.constant_initializer(self._init_std_param)
                    std_network = parameter(
                        state_input,
                        length=action_dim,
                        initializer=p,
                        trainable=self._learn_std,
                        name="std_network")

            mean_var = mean_network
            std_param_var = std_network

            with tf.variable_scope("std_limits"):
                if self._min_std_param:
                    std_param_var = tf.maximum(std_param_var,
                                               self._min_std_param)
                if self._max_std_param:
                    std_param_var = tf.minimum(std_param_var,
                                               self._max_std_param)

        with tf.variable_scope("std_parameterization"):
            # build std_var with std parameterization
            if self._std_parameterization == "exp":
                std_var = tf.exp(std_param_var)
            elif self._std_parameterization == "softplus":
                std_var = tf.log(1. + tf.exp(std_param_var))
            else:
                raise NotImplementedError

        dist = tf.contrib.distributions.MultivariateNormalDiag(
            mean_var, std_var)

        action_var = dist.sample(seed=ext.get_seed())

        return action_var, mean_var, std_param_var, dist
