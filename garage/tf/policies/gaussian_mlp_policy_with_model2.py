"""GaussianMLPPolicy with GaussianMLPModel."""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from garage.misc import logger
from garage.tf.models.gaussian_mlp_model2 import GaussianMLPModel2
from garage.tf.spaces import Box


class GaussianMLPPolicyWithModel2:
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
                 dist=tfp.distributions.MultivariateNormalDiag,
                 *args,
                 **kwargs):
        assert isinstance(env_spec.action_space, Box)

        self.name = name
        self._env_spec = env_spec
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        state_input = tf.placeholder(tf.float32, shape=(None, obs_dim))
        self.model = GaussianMLPModel2(output_dim=action_dim, *args, **kwargs)

        # There are two options:
        # 1. We can enforce the user to pass the input and build the model
        # once manually.
        # 2. We can build it by default.
        self.dist_info_sym(state_input, dist)

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True

    def dist_info_sym(self,
                      obs_var,
                      state_info_vars=None,
                      dist=tfp.distributions.MultivariateNormalDiag,
                      name=None):
        """Symbolic graph of the distribution."""
        with tf.variable_scope(self.name):
            _, _, dist = self.model.build(
                obs_var, tfp.distributions.MultivariateNormalDiag, name=name)
        return dist

    def get_action(self, observation):
        """Get action from the policy."""
        action = tf.get_default_session().run(
            self.model.networks['default'].sample,
            feed_dict={self.model.networks['default'].input: observation})
        return action

    def get_actions(self, observations):
        """Get actions from the policy."""
        actions = tf.get_default_session().run(
            self.model.networks['default'].sample,
            feed_dict={self.model.networks['default'].input: observations})
        return actions

    def get_reparam_action_sym(self,
                               obs_var,
                               action_var,
                               old_dist_info_vars,
                               name=None):
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
        if not name:
            name = 'get_reparam_action_sym'
        new_dist_info_vars = self.dist_info_sym(obs_var, name=name)
        new_mean_var, new_log_std_var = new_dist_info_vars[
            'mean'], new_dist_info_vars['log_std']
        old_mean_var, old_log_std_var = old_dist_info_vars[
            'mean'], old_dist_info_vars['log_std']
        epsilon_var = (action_var - old_mean_var) / (
            tf.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * tf.exp(new_log_std_var)
        return new_action_var

    def log_diagnostics(self, paths):
        """Log diagnostics."""
        log_stds = np.vstack(
            [path['agent_infos']['log_std'] for path in paths])
        logger.record_tabular('{}/AverageStd'.format(self.name),
                              np.mean(np.exp(log_stds)))
