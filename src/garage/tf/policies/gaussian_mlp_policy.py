"""Gaussian MLP Policy.

A policy represented by a Gaussian distribution
which is parameterized by a multilayer perceptron (MLP).
"""
# pylint: disable=wrong-import-order
import akro
import numpy as np
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.models import GaussianMLPModel
from garage.tf.policies.policy import Policy


# pylint: disable=too-many-ancestors
class GaussianMLPPolicy(GaussianMLPModel, Policy):
    """Gaussian MLP Policy.

    A policy represented by a Gaussian distribution
    which is parameterized by a multilayer perceptron (MLP).

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        name (str): Model name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
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
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network. The function should return a tf.Tensor.
        std_output_nonlinearity (callable): Nonlinearity for output layer in
            the std network. The function should return a
            tf.Tensor.
        std_parameterization (str): How the std should be parametrized. There
            are a few options:
        - exp: the logarithm of the std will be stored, and applied a
            exponential transformation
        - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 env_spec,
                 name='GaussianMLPPolicy',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 output_b_init=tf.zeros_initializer(),
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
        if not isinstance(env_spec.action_space, akro.Box):
            raise ValueError('GaussianMLPPolicy only works with '
                             'akro.Box action space, but not {}'.format(
                                 env_spec.action_space))

        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._learn_std = learn_std
        self._adaptive_std = adaptive_std
        self._std_share_network = std_share_network
        self._init_std = init_std
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_sizes = std_hidden_sizes
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_parameterization = std_parameterization
        self._layer_normalization = layer_normalization

        self._f_dist = None

        super().__init__(output_dim=self._action_dim,
                         hidden_sizes=hidden_sizes,
                         hidden_nonlinearity=hidden_nonlinearity,
                         hidden_w_init=hidden_w_init,
                         hidden_b_init=hidden_b_init,
                         output_nonlinearity=output_nonlinearity,
                         output_w_init=output_w_init,
                         output_b_init=output_b_init,
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
                         name=name)

        self._initialize_policy()

    def _initialize_policy(self):
        """Initialize policy."""
        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None,
                                                      self._obs_dim))
        dist, mean, log_std = self.build(state_input).outputs
        self._f_dist = tf.compat.v1.get_default_session().make_callable(
            [
                dist.sample(seed=deterministic.get_tf_seed_stream()), mean,
                log_std
            ],
            feed_list=[state_input])

    @property
    def input_dim(self):
        """int: Dimension of the policy input."""
        return self._obs_dim

    def get_action(self, observation):
        """Get single action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            numpy.ndarray: Actions
            dict: Predicted action and agent information.

        Note:
            It returns an action and a dict, with keys
            - mean (numpy.ndarray): Mean of the distribution.
            - log_std (numpy.ndarray): Log standard deviation of the
                distribution.

        """
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        """Get multiple actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment.

        Returns:
            numpy.ndarray: Actions
            dict: Predicted action and agent information.

        Note:
            It returns actions and a dict, with keys
            - mean (numpy.ndarray): Means of the distribution.
            - log_std (numpy.ndarray): Log standard deviations of the
                distribution.

        """
        if not isinstance(observations[0], np.ndarray):
            observations = self.observation_space.flatten_n(observations)
        samples, means, log_stds = self._f_dist(np.expand_dims(
            observations, 1))
        samples = self.action_space.unflatten_n(np.squeeze(samples, 1))
        means = self.action_space.unflatten_n(np.squeeze(means, 1))
        log_stds = self.action_space.unflatten_n(np.squeeze(log_stds, 1))
        return samples, dict(mean=means, log_std=log_stds)

    def clone(self, name):
        """Return a clone of the policy.

        It copies the configuration of the primitive and also the parameters.

        Args:
            name (str): Name of the newly created policy. It has to be
                different from source policy if cloned under the same
                computational graph.

        Returns:
            garage.tf.policies.GaussianMLPPolicy: Newly cloned policy.

        """
        new_policy = self.__class__(
            name=name,
            env_spec=self._env_spec,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            learn_std=self._learn_std,
            adaptive_std=self._adaptive_std,
            std_share_network=self._std_share_network,
            init_std=self._init_std,
            min_std=self._min_std,
            max_std=self._max_std,
            std_hidden_sizes=self._std_hidden_sizes,
            std_hidden_nonlinearity=self._std_hidden_nonlinearity,
            std_output_nonlinearity=self._std_output_nonlinearity,
            std_parameterization=self._std_parameterization,
            layer_normalization=self._layer_normalization)
        new_policy.parameters = self.parameters
        return new_policy

    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """
        return self._env_spec

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_dist']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize_policy()
