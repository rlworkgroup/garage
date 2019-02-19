"""GaussianMLPPolicy with GaussianMLPModel."""
import numpy as np
import tensorflow as tf

from garage.misc import logger
from garage.tf.distributions import DiagonalGaussian
from garage.tf.models.gaussian_mlp_model import GaussianMLPModel
from garage.tf.spaces import Box


class GaussianMLPPolicyWithModel:
    """
    GaussianMLPPolicy with GaussianMLPModel.

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

    def __init__(self,
                 env_spec,
                 name='GaussianMLPPolicy',
                 dist=DiagonalGaussian,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 output_nonlinearity=None,
                 learn_std=True,
                 adaptive_std=False,
                 std_share_network=False,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=tf.nn.tanh,
                 std_output_nonlinearity=None,
                 std_parameterization='exp'):
        assert isinstance(env_spec.action_space, Box)

        self.name = name
        self._env_spec = env_spec
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        state_input = tf.placeholder(tf.float32, shape=(None, obs_dim))
        self.model = GaussianMLPModel(
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
            learn_std=learn_std,
            adaptive_std=adaptive_std,
            std_share_network=std_share_network,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_hidden_sizes=std_hidden_sizes,
            std_hidden_nonlinearity=std_hidden_nonlinearity,
            std_output_nonlinearity=std_output_nonlinearity,
            std_parameterization=std_parameterization)

        with tf.variable_scope(name, reuse=False):
            _, self._mean_var, self._log_std_var, _, self._dist = \
                self.model.build(state_input, dist)

            self._f_dist = lambda x: tf.get_default_session().run(
                [self._mean_var, self._log_std_var],
                feed_dict={state_input: x})

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True

    def dist_info_sym(self,
                      obs_var,
                      state_info_vars=None,
                      name='dist_info_sym'):
        """Symbolic graph of the distribution."""
        with tf.variable_scope(
                self.name, reuse=True, auxiliary_name_scope=False):
            _, mean_var, log_std_var, _, _ = self.model.build(
                obs_var, DiagonalGaussian, name=name)
        return dict(mean=mean_var, log_std=log_std_var)

    def get_action(self, observation):
        """Get action from the policy."""
        flat_obs = self.observation_space.flatten(observation)
        mean, log_std = [x[0] for x in self._f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        """Get actions from the policy."""
        flat_obs = self.observation_space.flatten_n(observations)
        means, log_stds = self._f_dist(flat_obs)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def get_reparam_action_sym(self,
                               obs_var,
                               action_var,
                               old_dist_info_vars,
                               name="get_reparam_action_sym"):
        """
        Get symbolically reparamterzied action represnetation.

        Given observations, old actions, and distribution of old actions,
        return a symbolically reparameterized representation of the actions in
        terms of the policy parameters.

        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        new_dist_info_vars = self.dist_info_sym(obs_var, name=name)
        new_mean_var, new_log_std_var = new_dist_info_vars[
            "mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars[
            "mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (
            tf.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * tf.exp(new_log_std_var)
        return new_action_var

    def log_diagnostics(self, paths):
        """Log diagnostics."""
        log_stds = np.vstack(
            [path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular("{}/AverageStd".format(self.name),
                              np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        """Policy distribution."""
        return self._dist
