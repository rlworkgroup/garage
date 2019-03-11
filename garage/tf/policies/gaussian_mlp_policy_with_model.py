"""GaussianMLPPolicy with GaussianMLPModel."""
from akro.tf import Box
import numpy as np
import tensorflow as tf

from garage.tf.distributions import DiagonalGaussian
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
                 std_parameterization='exp',
                 layer_normalization=False):
        assert isinstance(env_spec.action_space, Box)
        super().__init__(name, env_spec)
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim

        self.model = GaussianMLPModel(
            name=name,
            output_dim=self.action_dim,
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
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization)

        self._initialize()

    def _initialize(self):
        state_input = tf.placeholder(tf.float32, shape=(None, self.obs_dim))

        with tf.variable_scope(self._variable_scope):
            self.model.build(state_input)

        self._f_dist = tf.get_default_session().make_callable(
            [
                self.model.networks['default'].output,
                self.model.networks['default'].std
            ],
            feed_list=[self.model.networks['default'].input])

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True

    def dist_info_sym(self, obs_var, state_info_vars=None, name='default'):
        """Symbolic graph of the distribution."""
        with tf.variable_scope(self._variable_scope):
            _, mean_var, log_std_var, _ = self.model.build(obs_var, name=name)
        return dict(mean=mean_var, log_std=log_std_var)

    def get_action(self, observation):
        """Get action from the policy."""
        flat_obs = self.observation_space.flatten(observation)
        mean, log_std = self._f_dist([flat_obs])
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

    def get_params(self, trainable=True):
        """Get the trainable variables."""
        return self.get_trainable_vars()

    @property
    def distribution(self):
        """Policy distribution."""
        return DiagonalGaussian(self.action_dim)

    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = self.__dict__.copy()
        del new_dict['_f_dist']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        self.__dict__.update(state)
        self._initialize()
