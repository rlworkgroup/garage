"""Gaussian MLP Model."""
import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.misc import ext
from garage.misc.overrides import overrides
from garage.tf.core.networks import mlp, parameter
from garage.tf.models import Model


class GaussianMLPModel(Model, Serializable):
    """Gaussian MLP Model."""

    def __init__(
            self,
            input_dim,
            output_dim,
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
    ):
        """
        Initialize a gaussian mlp model and build the graph.

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
        Serializable.quick_init(self, locals())
        super(GaussianMLPModel, self).__init__()

        self.name = name
        self._name_scope = tf.name_scope(self.name)

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

        inputs, outputs, model_info = self.build_model()
        self._inputs = inputs["input_var"]

        self.mean = outputs["mean"]
        self.std = outputs["std"]
        self.std_param = outputs["std_param"]
        self.sample = outputs["sample"]
        self.dist = model_info["dist"]

        self._outputs = outputs

    @overrides
    def build_model(self, inputs=None, reuse=tf.AUTO_REUSE):
        """
        Build the graph.

        return:
            inputs: a dict that contains input tensor
            outputs: a dict that contains mean,
                std, parameterized std and sample tensors
            model_info: a dict that contains the distribution tensor
        """
        if inputs is None:
            input_var = tf.placeholder(
                shape=[None, self._input_dim],
                dtype=tf.float32,
            )
        else:
            input_var = inputs

        with tf.variable_scope(self.name, reuse=reuse):
            if self._std_share_network:
                # mean and std networks share an MLP
                b = np.concatenate(
                    [
                        np.zeros(self._output_dim),
                        np.full(self._output_dim, self._init_std_param)
                    ],
                    axis=0)
                b = tf.constant_initializer(b)
                mean_std_network = mlp(
                    input_var=input_var,
                    output_dim=self._output_dim * 2,
                    hidden_sizes=self._hidden_sizes,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    output_nonlinearity=self._output_nonlinearity,
                    output_b_init=b,
                    name="mean_std_network")

                with tf.variable_scope("mean_network"):
                    mean_network = mean_std_network[..., :self._output_dim]
                with tf.variable_scope("std_network"):
                    std_network = mean_std_network[..., self._output_dim:]
            else:
                mean_network = mlp(
                    input_var=input_var,
                    output_dim=self._output_dim,
                    hidden_sizes=self._hidden_sizes,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    output_nonlinearity=self._output_nonlinearity,
                    name="mean_network")

                if self._adaptive_std:
                    b = tf.constant_initializer(self._init_std_param)
                    std_network = mlp(
                        input_var=input_var,
                        output_dim=self._output_dim,
                        hidden_sizes=self._std_hidden_sizes,
                        hidden_nonlinearity=self._std_hidden_nonlinearity,
                        output_nonlinearity=self._output_nonlinearity,
                        output_b_init=b,
                        name="std_network")

                else:
                    p = tf.constant_initializer(self._init_std_param)
                    std_network = parameter(
                        input_var=input_var,
                        length=self._output_dim,
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
        sample_var = dist.sample(seed=ext.get_seed())

        inputs = {
            "input_var": input_var,
        }
        outputs = {
            "mean": mean_var,
            "std": std_var,
            "std_param": std_param_var,
            "sample": sample_var,
        }
        model_info = {
            "dist": dist,
        }

        return inputs, outputs, model_info
