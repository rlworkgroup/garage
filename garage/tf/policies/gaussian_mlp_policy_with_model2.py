"""GaussianMLPPolicy with GaussianMLPModel2"""
from akro.tf import Box
import tensorflow as tf

from garage.tf.models.gaussian_mlp_model2 import GaussianMLPModel2


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
        self.model = GaussianMLPModel2(
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

        self.dist_info_sym(state_input)

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True

    def likelihood_ratio_sym(self, obs_var, other_dist):
        """Interface for likelihood ratio with another distribution."""
        assert (obs_var.shape.as_list() == self.model.networks['default'].
                input.shape.as_list())
        log_prob_diff = self.model.networks['default'].dist.log_prob(
            obs_var) - other_dist.log_prob(obs_var)

        return tf.exp(log_prob_diff)

    def dist_info_sym(self, obs_var, state_info_vars=None, name=None):
        """Symbolic graph of the distribution."""
        _, _, dist = self.model.build(obs_var, name=name)
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
