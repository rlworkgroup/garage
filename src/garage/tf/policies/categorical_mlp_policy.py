"""CategoricalMLPPolicy."""
import akro
import tensorflow as tf

from garage.tf.distributions import Categorical
from garage.tf.models import MLPModel
from garage.tf.policies import StochasticPolicy


class CategoricalMLPPolicy(StochasticPolicy):
    """Estimate action distribution with Categorical parameterized by a MLP.

    A policy that contains a MLP to make prediction based on
    a categorical distribution.

    It only works with akro.Discrete action space.

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
                 name='CategoricalMLPPolicy',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=tf.nn.softmax,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 layer_normalization=False):
        assert isinstance(env_spec.action_space, akro.Discrete), (
            'CategoricalMLPPolicy only works with akro.Discrete action '
            'space.')
        super().__init__(name, env_spec)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.n
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization

        self.model = MLPModel(output_dim=self._action_dim,
                              hidden_sizes=hidden_sizes,
                              hidden_nonlinearity=hidden_nonlinearity,
                              hidden_w_init=hidden_w_init,
                              hidden_b_init=hidden_b_init,
                              output_nonlinearity=output_nonlinearity,
                              output_w_init=output_w_init,
                              output_b_init=output_b_init,
                              layer_normalization=layer_normalization,
                              name='MLPModel')

        self._initialize()

    def _initialize(self):
        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, self._obs_dim))

        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            self.model.build(state_input)

        self._f_prob = tf.compat.v1.get_default_session().make_callable(
            self.model.networks['default'].outputs,
            feed_list=[self.model.networks['default'].input])

    @property
    def vectorized(self):
        """Vectorized or not.

        Returns:
            bool: True if primitive supports vectorized operations.

        """
        return True

    def dist_info_sym(self, obs_var, state_info_vars=None, name=None):
        """Build a symbolic graph of the distribution parameters.

        Args:
            obs_var (tf.Tensor): Tensor input for symbolic graph.
            state_info_vars (dict[tf.Tensor]): Extra state information, e.g.
                previous action.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of distribution
                parameters.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            prob = self.model.build(obs_var, name=name)
        return dict(prob=prob)

    def dist_info(self, obs, state_infos=None):
        """Build a symbolic graph of the distribution parameters.

        Args:
            obs (np.ndarray): Input for symbolic graph.
            state_infos (dict[np.ndarray]): Extra state information, e.g.
                previous action.

        Returns:
            dict[np.ndarray]: Outputs of the symbolic graph of distribution
                parameters.

        """
        prob = self._f_prob(obs)
        return dict(prob=prob)

    def get_action(self, observation):
        """Get single action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            numpy.ndarray: Predicted action.
            dict[str: np.ndarray]: Action distribution.

        """
        flat_obs = self.observation_space.flatten(observation)
        prob = self._f_prob([flat_obs])[0]
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        """Get multiple actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment.

        Returns:
            numpy.ndarray: Predicted actions.
            dict[str: np.ndarray]: Action distributions.

        """
        flat_obs = self.observation_space.flatten_n(observations)
        probs = self._f_prob(flat_obs)
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

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
    def distribution(self):
        """Policy distribution.

        Returns:
            garage.tf.distributions.Categorical: Policy distribution.

        """
        return Categorical(self._action_dim)

    def clone(self, name):
        """Return a clone of the policy.

        It only copies the configuration of the Q-function,
        not the parameters.

        Args:
            name (str): Name of the newly created policy.

        Returns:
            garage.tf.policies.CategoricalMLPPolicy: Clone of this object

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
                              layer_normalization=self._layer_normalization)

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_prob']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
