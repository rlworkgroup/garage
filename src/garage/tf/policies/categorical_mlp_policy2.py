"""CategoricalMLPPolicy with model."""
import akro
import tensorflow as tf
import tensorflow_probability as tfp

from garage.misc.overrides import overrides
from garage.tf.models import MLPModel
from garage.tf.policies.base import StochasticPolicy2


class CategoricalMLPPolicy2(StochasticPolicy2):
    """CategoricalMLPPolicy with model.

    A policy that contains a MLP to make prediction based on
    a tfp.distributions.OneHotCategorical.

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
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.n

        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization

        self.model = MLPModel(output_dim=self.action_dim,
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
                                               shape=(None, self.obs_dim))

        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            prob = self.model.build(state_input)

        self._dist = tfp.distributions.OneHotCategorical(prob)

        self._f_prob = tf.compat.v1.get_default_session().make_callable(
            prob, feed_list=self.model.networks['default'].inputs)

    @property
    def vectorized(self):
        """Vectorized or not.

        Returns:
            Bool: True if primitive supports vectorized operations.

        """
        return True

    @overrides
    def dist_info(self, obs, state_infos=None):
        """Distribution info.

        Args:
            obs (numpy.ndarray): Observations.
            state_infos (dict): Extra state information.

        Returns:
            numpy.ndarray: Distribution info given the input observations.

        """
        return self._f_prob(obs)

    @overrides
    def get_action(self, observation):
        """Return a single action.

        Args:
            observation (numpy.ndarray): Observations.

        Returns:
            int: Action given input observation.
            dict(numpy.ndarray): Distribution parameters.

        """
        flat_obs = self.observation_space.flatten(observation)
        prob = self._f_prob([flat_obs])[0]
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        """Return multiple actions.

        Args:
            observations (numpy.ndarray): Observations.

        Returns:
            list[int]: Actions given input observations.
            dict(numpy.ndarray): Distribution parameters.

        """
        flat_obs = self.observation_space.flatten_n(observations)
        probs = self._f_prob(flat_obs)
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

    def get_regularizable_vars(self):
        """Get regularizable weight variables under the Policy scope.

        Returns:
            list[tf.Tensor]: Trainable variables.

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
            tfp.Distribution: Policy distribution.

        """
        return self._dist

    def clone(self, name):
        """Return a clone of the policy.

        It only copies the configuration of the primitive,
        not the parameters.

        Args:
            name (str): Name of the newly created policy. It has to be
                different from source policy if cloned under the same
                computational graph.

        Returns:
            garage.tf.policies.Policy: Newly cloned policy.

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
            dict: State dictionary.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_prob']
        del new_dict['_dist']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): State dictionary.

        """
        super().__setstate__(state)
        self._initialize()
