"""GaussianMLPPolicy with GaussianMLPModel."""
from akro.tf import Box
import numpy as np
import tensorflow as tf

from garage.tf.misc import tensor_utils
from garage.tf.models.gaussian_mlp_model import GaussianMLPModel
from garage.tf.policies.base2 import StochasticPolicy2


class GaussianMLPPolicyWithModel(StochasticPolicy2):
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
            name=name,
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

        self.model.build(state_input)

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True

    def dist_info_sym(self, obs_var, state_info_vars=None, name='default'):
        """Symbolic graph of the distribution."""
        _, mean_var, log_std_var, _, _ = self.model.build(obs_var, name=name)
        return dict(mean=mean_var, log_std=log_std_var)

    def get_action(self, observation):
        """Get action from the policy."""
        flat_obs = self.observation_space.flatten(observation)
        _f_dist = tensor_utils.compile_function(
            inputs=[self.model.networks['default'].input],
            outputs=[
                self.model.networks['default'].mean,
                self.model.networks['default'].std
            ])
        mean, log_std = [x[0] for x in _f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        """Get actions from the policy."""
        flat_obs = self.observation_space.flatten_n(observations)
        _f_dist = tensor_utils.compile_function(
            inputs=[self.model.networks['default'].input],
            outputs=[
                self.model.networks['default'].mean,
                self.model.networks['default'].std
            ])
        means, log_stds = _f_dist(flat_obs)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def get_params(self, trainable=True):
        """Get the trainable variables."""
        return self.model._variable_scope.trainable_variables()

    @property
    def distribution(self):
        """Policy distribution."""
        return self.model.networks['default'].dist
