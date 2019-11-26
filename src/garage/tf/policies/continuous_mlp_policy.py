"""This modules creates a continuous MLP policy network.

A continuous MLP network can be used as policy method in different RL
algorithms. It accepts an observation of the environment and predicts a
continuous action.
"""
import tensorflow as tf

from garage.tf.models import MLPModel
from garage.tf.policies import Policy


class ContinuousMLPPolicy(Policy):
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
        input_include_goal (bool): Include goal in the observation or not.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 env_spec,
                 name='ContinuousMLPPolicy',
                 hidden_sizes=(64, 64),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=tf.nn.tanh,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 input_include_goal=False,
                 layer_normalization=False):
        super().__init__(name, env_spec)
        action_dim = env_spec.action_space.flat_dim
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._input_include_goal = input_include_goal
        self._layer_normalization = layer_normalization
        if self._input_include_goal:
            self.obs_dim = env_spec.observation_space.flat_dim_with_keys(
                ['observation', 'desired_goal'])
        else:
            self.obs_dim = env_spec.observation_space.flat_dim

        self.model = MLPModel(output_dim=action_dim,
                              name='MLPModel',
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
                                               shape=(None, self.obs_dim))

        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            self.model.build(state_input)

        self._f_prob = tf.compat.v1.get_default_session().make_callable(
            self.model.networks['default'].outputs,
            feed_list=[self.model.networks['default'].input])

    def get_action_sym(self, obs_var, name=None):
        """Symbolic graph of the action.

        Args:
            obs_var (tf.Tensor): Tensor input for symbolic graph.
            name (str): Name for symbolic graph.

        Returns:
            tf.Tensor: symbolic graph of the action.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            return self.model.build(obs_var, name=name)

    def get_action(self, observation):
        """Get single action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            numpy.ndarray: Predicted action.
            dict: Empty dict since this policy does not model a distribution.

        """
        action = self._f_prob([observation])
        action = self.action_space.unflatten(action)
        return action, dict()

    def get_actions(self, observations):
        """Get multiple actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment.

        Returns:
            numpy.ndarray: Predicted actions.
            dict: Empty dict since this policy does not model a distribution.

        """
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
    def vectorized(self):
        """Vectorized or not.

        Returns:
            bool: vectorized or not.

        """
        return True

    def clone(self, name):
        """Return a clone of the policy.

        It only copies the configuration of the Q-function,
        not the parameters.

        Args:
            name (str): Name of the newly created policy.

        Returns:
            garage.tf.policies.ContinuousMLPPolicy: Clone of this object

        """
        return self.__class__(name=name,
                              env_spec=self._env_spec,
                              hidden_sizes=self._hidden_sizes,
                              hidden_nonlinearity=self._hidden_nonlinearity,
                              hidden_w_init=self._hidden_w_init,
                              hidden_b_init=self._hidden_b_init,
                              output_nonlinearity=self._output_nonlinearity,
                              output_w_init=self._output_w_init,
                              output_b_init=self._output_b_init,
                              input_include_goal=self._input_include_goal,
                              layer_normalization=self._layer_normalization)

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
