"""This modules creates a continuous MLP policy network.

A continuous MLP network can be used as policy method in different RL
algorithms. It accepts an observation of the environment and predicts a
continuous action.
"""
import numpy as np
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.models import MLPModel
from garage.tf.policies.policy import Policy


# pylint: disable=too-many-ancestors
class ContinuousMLPPolicy(MLPModel, Policy):
    """Continuous MLP Policy Network.

    The policy network selects action based on the state of the environment.
    It uses neural nets to fit the function of pi(s).

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        name (str): Policy name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means the MLP of this policy consists of two
            hidden layers, each with 32 hidden units.
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
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 env_spec,
                 name='ContinuousMLPPolicy',
                 hidden_sizes=(64, 64),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=tf.nn.tanh,
                 output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 output_b_init=tf.zeros_initializer(),
                 layer_normalization=False):
        self._env_spec = env_spec
        action_dim = env_spec.action_space.flat_dim
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        self._obs_dim = env_spec.observation_space.flat_dim

        super().__init__(output_dim=action_dim,
                         name=name,
                         hidden_sizes=hidden_sizes,
                         hidden_nonlinearity=hidden_nonlinearity,
                         hidden_w_init=hidden_w_init,
                         hidden_b_init=hidden_b_init,
                         output_nonlinearity=output_nonlinearity,
                         output_w_init=output_w_init,
                         output_b_init=output_b_init,
                         layer_normalization=layer_normalization)

        self._initialize()

    def _initialize(self):
        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, self._obs_dim))
        outputs = super().build(state_input).outputs

        self._f_prob = tf.compat.v1.get_default_session().make_callable(
            outputs, feed_list=[state_input])

    # pylint: disable=arguments-differ
    def build(self, obs_var, name=None):
        """Symbolic graph of the action.

        Args:
            obs_var (tf.Tensor): Tensor input for symbolic graph.
            name (str): Name for symbolic graph.

        Returns:
            tf.Tensor: symbolic graph of the action.

        """
        return super().build(obs_var, name=name).outputs

    @property
    def input_dim(self):
        """int: Dimension of the policy input."""
        return self._obs_dim

    def get_action(self, observation):
        """Get single action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            numpy.ndarray: Predicted action.
            dict: Empty dict since this policy does not model a distribution.

        """
        actions, agent_infos = self.get_actions([observation])
        action = actions[0]
        return action, {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        """Get multiple actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment.

        Returns:
            numpy.ndarray: Predicted actions.
            dict: Empty dict since this policy does not model a distribution.

        """
        if not isinstance(observations[0], np.ndarray):
            observations = self.observation_space.flatten_n(observations)
        actions = self._f_prob(observations)
        actions = self.action_space.unflatten_n(actions)
        return actions, dict()

    def get_regularizable_vars(self):
        """Get regularizable weight variables under the Policy scope.

        Returns:
            list(tf.Variable): List of regularizable variables.

        """
        trainable = self.get_trainable_vars()
        return [
            var for var in trainable
            if 'hidden' in var.name and 'kernel' in var.name
        ]

    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """
        return self._env_spec

    def clone(self, name):
        """Return a clone of the policy.

        It copies the configuration of the primitive and also the parameters.

        Args:
            name (str): Name of the newly created policy.

        Returns:
            garage.tf.policies.ContinuousMLPPolicy: Clone of this object

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
            layer_normalization=self._layer_normalization)
        new_policy.parameters = self.parameters
        return new_policy

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled as the contents for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_prob']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
