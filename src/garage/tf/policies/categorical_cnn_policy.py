"""Categorical CNN Policy.

A policy represented by a Categorical distribution
which is parameterized by a convolutional neural network (CNN)
followed a multilayer perceptron (MLP).
"""
# pylint: disable=wrong-import-order
import akro
import numpy as np
import tensorflow as tf

from garage.tf.models import CategoricalCNNModel
from garage.tf.policies.policy import StochasticPolicy


class CategoricalCNNPolicy(StochasticPolicy):
    """CategoricalCNNPolicy.

    A policy that contains a CNN and a MLP to make prediction based on
    a categorical distribution.

    It only works with akro.Discrete action space.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        filters (Tuple[Tuple[int, Tuple[int, int]], ...]): Number and dimension
            of filters. For example, ((3, (3, 5)), (32, (3, 3))) means there
            are two convolutional layers. The filter for the first layer have 3
            channels and its shape is (3 x 5), while the filter for the second
            layer have 32 channels and its shape is (3 x 3).
        strides (tuple[int]): The stride of the sliding window. For
            example, (1, 2) means there are two convolutional layers. The
            stride of the filter for first layer is 1 and that of the second
            layer is 2.
        padding (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        name (str): Policy name, also the variable scope of the policy.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means the MLP of this policy consists
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
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 env_spec,
                 filters,
                 strides,
                 padding,
                 name='CategoricalCNNPolicy',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.initializers.glorot_uniform(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(),
                 output_b_init=tf.zeros_initializer(),
                 layer_normalization=False):
        assert isinstance(env_spec.action_space, akro.Discrete), (
            'CategoricalCNNPolicy only works with akro.Discrete action '
            'space.')
        super().__init__(name, env_spec)
        self._obs_dim = env_spec.observation_space.shape
        self._action_dim = env_spec.action_space.n
        self._filters = filters
        self._strides = strides
        self._padding = padding
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization

        self._f_prob = None
        self._dist = None

        self.model = CategoricalCNNModel(
            output_dim=self._action_dim,
            filters=filters,
            strides=strides,
            padding=padding,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

    def build(self, state_input, name=None):
        """Build model.

        Args:
          state_input (tf.Tensor) : State input.
          name (str): Name of the model, which is also the name scope.

        """
        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            if isinstance(self.env_spec.observation_space, akro.Image):
                augmented_state_input = tf.cast(state_input, tf.float32)
                augmented_state_input /= 255.0
            else:
                augmented_state_input = state_input
            self._dist = self.model.build(augmented_state_input, name=name)
            self._f_prob = tf.compat.v1.get_default_session().make_callable(
                [tf.argmax(self._dist.sample(), -1), self._dist.probs],
                feed_list=[state_input])

    @property
    def distribution(self):
        """Policy distribution.

        Returns:
            tfp.Distribution.OneHotCategorical: Policy distribution.

        """
        return self._dist

    @property
    def vectorized(self):
        """Vectorized or not.

        Returns:
            bool: True if primitive supports vectorized operations.

        """
        return True

    def get_action(self, observation):
        """Return a single action.

        Args:
            observation (numpy.ndarray): Observations.

        Returns:
            int: Action given input observation.
            dict(numpy.ndarray): Distribution parameters.

        """
        sample, prob = self._f_prob(np.expand_dims([observation], 1))
        return np.squeeze(sample), dict(prob=np.squeeze(prob, axis=1)[0])

    def get_actions(self, observations):
        """Return multiple actions.

        Args:
            observations (numpy.ndarray): Observations.

        Returns:
            list[int]: Actions given input observations.
            dict(numpy.ndarray): Distribution parameters.

        """
        samples, probs = self._f_prob(np.expand_dims(observations, 1))
        return np.squeeze(samples), dict(prob=np.squeeze(probs, axis=1))

    def clone(self, name):
        """Return a clone of the policy.

        It only copies the configuration of the primitive,
        not the parameters.

        Args:
            name (str): Name of the newly created policy. It has to be
                different from source policy if cloned under the same
                computational graph.

        Returns:
            garage.tf.policies.CategoricalCNNPolicy: Newly cloned policy.

        """
        return self.__class__(name=name,
                              env_spec=self._env_spec,
                              filters=self._filters,
                              strides=self._strides,
                              padding=self._padding,
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
            dict: The state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_prob']
        del new_dict['_dist']
        return new_dict
