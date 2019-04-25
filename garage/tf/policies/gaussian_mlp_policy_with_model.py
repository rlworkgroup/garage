"""GaussianMLPPolicy with GaussianMLPModel."""
from akro.tf import Box
import tensorflow as tf

from garage.tf.models import GaussianMLPModel
from garage.tf.policies.base2 import StochasticPolicy2


class GaussianMLPPolicyWithModel(StochasticPolicy2):
    """
    GaussianMLPPolicy with GaussianMLPModel.

    A policy that contains a MLP to make prediction based on
    a gaussian distribution.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        name (str): Model name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity: Nonlinearity used for each hidden layer.
        output_nonlinearity: Nonlinearity for the output layer.
        learn_std (bool): Is std trainable.
        adaptive_std (bool): Is std a neural network. If False, it will be a
            parameter.
        std_share_network (bool): Boolean for whether mean and std share
            the same network.
        init_std (float): Initial value for std.
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues.
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues.
        std_hidden_nonlinearity: Nonlinearity for each hidden layer in
            the std network.
        std_output_nonlinearity: Nonlinearity for output layer in
            the std network.
        std_parametrization (str): How the std should be parametrized. There
            are a few options:
        - exp: the logarithm of the std will be stored, and applied a
            exponential transformation
        - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
    :return:

    """

    def __init__(self,
                 env_spec,
                 name='GaussianMLPPolicyWithModel',
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
            layer_normalization=layer_normalization,
            name='GaussianMLPModel')

        self._initialize()

    def _initialize(self):
        state_input = tf.placeholder(tf.float32, shape=(None, self.obs_dim))

        with tf.variable_scope(self._variable_scope):
            self.model.build(state_input)

        self._f_dist = tf.get_default_session().make_callable(
            [
                self.model.networks['default'].sample,
                self.model.networks['default'].mean,
                self.model.networks['default'].log_std
            ],
            feed_list=[self.model.networks['default'].input])

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True

    def dist_info_sym(self, obs_var, state_info_vars=None, name='default'):
        """Symbolic graph of the distribution."""
        with tf.variable_scope(self._variable_scope):
            _, mean_var, log_std_var, _, _ = self.model.build(
                obs_var, name=name)
        return dict(mean=mean_var, log_std=log_std_var)

    def get_action(self, observation):
        """Get action from the policy."""
        flat_obs = self.observation_space.flatten(observation)
        sample, mean, log_std = self._f_dist([flat_obs])
        sample = self.action_space.unflatten(sample[0])
        mean = self.action_space.unflatten(mean[0])
        log_std = self.action_space.unflatten(log_std[0])
        return sample, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        """Get actions from the policy."""
        flat_obs = self.observation_space.flatten_n(observations)
        samples, means, log_stds = self._f_dist(flat_obs)
        samples = self.action_space.unflatten_n(samples)
        means = self.action_space.unflatten_n(means)
        log_stds = self.action_space.unflatten_n(log_stds)
        return samples, dict(mean=means, log_std=log_stds)

    def get_params(self, trainable=True):
        """Get the trainable variables."""
        return self.get_trainable_vars()

    @property
    def distribution(self):
        """Policy distribution."""
        return self.model.networks['default'].dist

    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = self.__dict__.copy()
        del new_dict['_f_dist']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        self.__dict__.update(state)
        self._initialize()
