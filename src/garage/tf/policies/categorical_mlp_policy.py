"""Categorical MLP Policy.

A policy represented by a Categorical distribution
which is parameterized by a multilayer perceptron (MLP).
"""
# pylint: disable=wrong-import-order
import akro
import numpy as np
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.models import CategoricalMLPModel
from garage.tf.policies.policy import StochasticPolicy


class CategoricalMLPPolicy(StochasticPolicy):
    """Categorical MLP Policy.

    A policy represented by a Categorical distribution
    which is parameterized by a multilayer perceptron (MLP).

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
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 output_b_init=tf.zeros_initializer(),
                 layer_normalization=False):
        if not isinstance(env_spec.action_space, akro.Discrete):
            raise ValueError('CategoricalMLPPolicy only works'
                             'with akro.Discrete action space.')
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

        self._f_prob = None
        self._dist = None

        self.model = CategoricalMLPModel(
            output_dim=self._action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization,
            name='CategoricalMLPModel')

        self._initialize()

    def _initialize(self):
        """Initialize policy."""
        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            state_input = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None, None,
                                                          self._obs_dim))
            self._dist = self.model.build(state_input).dist
            self._f_prob = tf.compat.v1.get_default_session().make_callable(
                [
                    tf.argmax(
                        self._dist.sample(
                            seed=deterministic.get_tf_seed_stream()), -1),
                    self._dist.probs
                ],
                feed_list=[state_input])

    def build(self, state_input, name=None):
        """Build policy.

        Args:
            state_input (tf.Tensor) : State input.
            name (str): Name of the policy, which is also the name scope.

        Returns:
            tfp.distributions.OneHotCategorical: Policy distribution.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            return self.model.build(state_input, name=name)

    @property
    def input_dim(self):
        """int: Dimension of the policy input."""
        return self._obs_dim

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
            Bool: True if primitive supports vectorized operations.

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
        actions, agent_infos = self.get_actions([observation])
        return actions, {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        """Return multiple actions.

        Args:
            observations (numpy.ndarray): Observations.

        Returns:
            list[int]: Actions given input observations.
            dict(numpy.ndarray): Distribution parameters.

        """
        observations = self.observation_space.flatten_n(observations)
        samples, probs = self._f_prob(np.expand_dims(observations, 1))
        return np.squeeze(samples), dict(prob=np.squeeze(probs, axis=1))

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

    def clone(self, name):
        """Return a clone of the policy.

        It copies the configuration of the primitive and also the parameters.

        Args:
            name (str): Name of the newly created policy. It has to be
                different from source policy if cloned under the same
                computational graph.

        Returns:
            garage.tf.policies.Policy: Newly cloned policy.

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
        new_policy.model.parameters = self.model.parameters
        return new_policy

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
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
