from akro.tf import Box
import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.logger import tabular
from garage.misc.overrides import overrides
from garage.tf.core import LayersPowered
import garage.tf.core.layers as L
from garage.tf.core.network import MLP
from garage.tf.distributions import DiagonalGaussian
from garage.tf.misc import tensor_utils
from garage.tf.policies.base import StochasticPolicy


class GaussianMLPPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(self,
                 env_spec,
                 name="GaussianMLPPolicy",
                 hidden_sizes=(32, 32),
                 learn_std=True,
                 init_std=1.0,
                 adaptive_std=False,
                 std_share_network=False,
                 std_hidden_sizes=(32, 32),
                 min_std=1e-6,
                 std_hidden_nonlinearity=tf.nn.tanh,
                 hidden_nonlinearity=tf.nn.tanh,
                 output_nonlinearity=None,
                 mean_network=None,
                 std_network=None,
                 std_parametrization='exp'):
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
        assert isinstance(env_spec.action_space, Box)

        Serializable.quick_init(self, locals())
        self.name = name
        self._mean_network_name = "mean_network"
        self._std_network_name = "std_network"

        with tf.variable_scope(name, "GaussianMLPPolicy"):

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            # create network
            if mean_network is None:
                if std_share_network:
                    if std_parametrization == "exp":
                        init_std_param = np.log(init_std)
                    elif std_parametrization == "softplus":
                        init_std_param = np.log(np.exp(init_std) - 1)
                    else:
                        raise NotImplementedError
                    b = np.concatenate((np.zeros(action_dim),
                                        np.full(action_dim, init_std_param)),
                                       axis=0)
                    b = tf.constant_initializer(b)
                    with tf.variable_scope(self._mean_network_name):
                        mean_network = MLP(
                            name="mlp",
                            input_shape=(obs_dim, ),
                            output_dim=2 * action_dim,
                            hidden_sizes=hidden_sizes,
                            hidden_nonlinearity=hidden_nonlinearity,
                            output_nonlinearity=output_nonlinearity,
                            output_b_init=b,
                        )
                        l_mean = L.SliceLayer(
                            mean_network.output_layer,
                            slice(action_dim),
                            name="mean_slice",
                        )
                else:
                    mean_network = MLP(
                        name=self._mean_network_name,
                        input_shape=(obs_dim, ),
                        output_dim=action_dim,
                        hidden_sizes=hidden_sizes,
                        hidden_nonlinearity=hidden_nonlinearity,
                        output_nonlinearity=output_nonlinearity,
                    )
                    l_mean = mean_network.output_layer
            self._mean_network = mean_network

            obs_var = mean_network.input_layer.input_var

            if std_network is not None:
                l_std_param = std_network.output_layer
            else:
                if std_parametrization == 'exp':
                    init_std_param = np.log(init_std)
                elif std_parametrization == 'softplus':
                    init_std_param = np.log(np.exp(init_std) - 1)
                else:
                    raise NotImplementedError
                if adaptive_std:
                    b = tf.constant_initializer(init_std_param)
                    std_network = MLP(
                        name=self._std_network_name,
                        input_shape=(obs_dim, ),
                        input_layer=mean_network.input_layer,
                        output_dim=action_dim,
                        hidden_sizes=std_hidden_sizes,
                        hidden_nonlinearity=std_hidden_nonlinearity,
                        output_nonlinearity=None,
                        output_b_init=b,
                    )
                    l_std_param = std_network.output_layer
                elif std_share_network:
                    with tf.variable_scope(self._std_network_name):
                        l_std_param = L.SliceLayer(
                            mean_network.output_layer,
                            slice(action_dim, 2 * action_dim),
                            name="std_slice",
                        )
                else:
                    with tf.variable_scope(self._std_network_name):
                        l_std_param = L.ParamLayer(
                            mean_network.input_layer,
                            num_units=action_dim,
                            param=tf.constant_initializer(init_std_param),
                            name="output_std_param",
                            trainable=learn_std,
                        )

            self.std_parametrization = std_parametrization

            if std_parametrization == 'exp':
                min_std_param = np.log(min_std)
            elif std_parametrization == 'softplus':
                min_std_param = np.log(np.exp(min_std) - 1)
            else:
                raise NotImplementedError

            self.min_std_param = min_std_param

            # mean_var, log_std_var = L.get_output([l_mean, l_std_param])
            #
            # if self.min_std_param is not None:
            #     log_std_var = tf.maximum(log_std_var, np.log(min_std))
            #
            # self._mean_var, self._log_std_var = mean_var, log_std_var

            self._l_mean = l_mean
            self._l_std_param = l_std_param

            self._dist = DiagonalGaussian(action_dim)

            LayersPowered.__init__(self, [l_mean, l_std_param])
            super(GaussianMLPPolicy, self).__init__(env_spec)

            dist_info_sym = self.dist_info_sym(
                mean_network.input_layer.input_var, dict())
            mean_var = tf.identity(dist_info_sym["mean"], name="mean")
            log_std_var = tf.identity(
                dist_info_sym["log_std"], name="standard_dev")

            self._f_dist = tensor_utils.compile_function(
                inputs=[obs_var],
                outputs=[mean_var, log_std_var],
            )

    @property
    def vectorized(self):
        return True

    def dist_info_sym(self, obs_var, state_info_vars=None, name=None):
        with tf.name_scope(name, "dist_info_sym", [obs_var]):
            with tf.name_scope(self._mean_network_name, values=[obs_var]):
                mean_var = L.get_output(self._l_mean, obs_var)
            with tf.name_scope(self._std_network_name, values=[obs_var]):
                std_param_var = L.get_output(self._l_std_param, obs_var)
            if self.min_std_param is not None:
                std_param_var = tf.maximum(std_param_var, self.min_std_param)
            if self.std_parametrization == 'exp':
                log_std_var = std_param_var
            elif self.std_parametrization == 'softplus':
                log_std_var = tf.log(tf.log(1. + tf.exp(std_param_var)))
            else:
                raise NotImplementedError
            return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        mean, log_std = [x[0] for x in self._f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        means, log_stds = self._f_dist(flat_obs)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def get_reparam_action_sym(self,
                               obs_var,
                               action_var,
                               old_dist_info_vars,
                               name=None):
        """
        Given observations, old actions, and distribution of old actions,
        return a symbolically reparameterized representation of the actions in
        terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        with tf.name_scope(name, "get_reparam_action_sym",
                           [obs_var, action_var, old_dist_info_vars]):
            new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
            new_mean_var, new_log_std_var = new_dist_info_vars[
                "mean"], new_dist_info_vars["log_std"]
            old_mean_var, old_log_std_var = old_dist_info_vars[
                "mean"], old_dist_info_vars["log_std"]
            epsilon_var = (action_var - old_mean_var) / (
                tf.exp(old_log_std_var) + 1e-8)
            new_action_var = new_mean_var + epsilon_var * tf.exp(
                new_log_std_var)
            return new_action_var

    def log_diagnostics(self, paths):
        log_stds = paths["agent_infos"]["log_std"]
        tabular.record("{}/AverageStd".format(self.name),
                              np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self._dist
