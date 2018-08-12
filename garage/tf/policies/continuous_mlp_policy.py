"""
This modules creates a continuous MLP policy network.

A continuous MLP network can be used as policy method in different RL
algorithms. It accepts an observation of the environment and predicts an
action.
"""
import tensorflow as tf

from garage.core import Serializable
from garage.envs.util import flat_dim
from garage.misc.overrides import overrides
from garage.tf.core import layers as layers
from garage.tf.core import LayersPowered
from garage.tf.core.layers import batch_norm
from garage.tf.misc import tensor_utils
from garage.tf.policies import Policy
from garage.tf.spaces import Box


class ContinuousMLPPolicy(Policy, Serializable, LayersPowered):
    """
    This class implements a policy network.

    The policy network selects action based on the state of the environment.
    It uses neural nets to fit the function of pi(s).
    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(64, 64),
                 name="ContinuousMLPPolicy",
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=tf.nn.tanh,
                 input_include_goal=False,
                 bn=False):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec():
            hidden_sizes(list or tuple, optional):
                A list of numbers of hidden units for all hidden layers.
            name(str, optional):
                A str contains the name of the policy.
            hidden_nonlinearity(optional):
                An activation shared by all fc layers.
            output_nonlinearity(optional):
                An activation used by the output layer.
            bn(bool, optional):
                A bool to indicate whether normalize the layer or not.
        """
        assert isinstance(env_spec.action_space, Box)

        Serializable.quick_init(self, locals())
        super(ContinuousMLPPolicy, self).__init__(env_spec)

        self.name = name
        self._env_spec = env_spec
        if input_include_goal:
            obs_dim = flat_dim(
                env_spec.observation_space.spaces["observation"])
            goal_dim = flat_dim(
                env_spec.observation_space.spaces["desired_goal"])
            self._obs_dim = obs_dim + goal_dim
        else:
            self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._action_bound = env_spec.action_space.high
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._batch_norm = bn
        self._policy_network_name = "policy_network"

    def _build_net(self, reuse=None, custom_getter=None, trainable=None):
        """
        Set up q network based on class attributes.

        This function uses layers defined in garage.tf.

        Args:
            reuse: A bool indicates whether reuse variables in the same scope.
            custom_getter: A customized getter object used to get variables.
            trainable: A bool indicates whether variables are trainable.
        """
        with tf.variable_scope(
                self.name, reuse=reuse, custom_getter=custom_getter):
            l_in = layers.InputLayer(shape=(None, self._obs_dim), name="obs")

            l_hidden = l_in
            for idx, hidden_size in enumerate(self._hidden_sizes):
                if self._batch_norm:
                    l_hidden = batch_norm(l_hidden)

                l_hidden = layers.DenseLayer(
                    l_hidden,
                    hidden_size,
                    nonlinearity=self._hidden_nonlinearity,
                    trainable=trainable,
                    name="hidden_%d" % idx)

            l_output = layers.DenseLayer(
                l_hidden,
                self._action_dim,
                nonlinearity=self._output_nonlinearity,
                trainable=trainable,
                name="output")

            with tf.name_scope(self._policy_network_name):
                action = layers.get_output(l_output)
                scaled_action = tf.multiply(
                    action, self._action_bound, name="scaled_action")

        self._f_prob_online = tensor_utils.compile_function(
            inputs=[l_in.input_var], outputs=scaled_action)
        self._output_layer = l_output
        self._obs_layer = l_in

        LayersPowered.__init__(self, [l_output])

    def get_action_sym(self, obs_var, name=None, **kwargs):
        """Return action sym according to obs_var."""
        with tf.name_scope(name, "get_action_sym", [obs_var]):
            with tf.name_scope(self._policy_network_name):
                actions = layers.get_output(
                    self._output_layer, {self._obs_layer: obs_var}, **kwargs)
            return tf.multiply(actions, self._action_bound)

    @overrides
    def get_action(self, observation):
        """Return a single action."""
        return self._f_prob_online([observation])[0], dict()

    @overrides
    def get_actions(self, observations):
        """Return multiple actions."""
        return self._f_prob_online(observations), dict()

    @property
    def trainable_vars(self):
        """Return trainable vars in the network."""
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def global_vars(self):
        """Return the global vars in the network."""
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def regularizable_vars(self):
        """Return regularizable vars in the network."""
        reg_vars = [
            var for var in self.trainable_vars
            if 'W' in var.name and 'output' not in var.name
        ]
        return reg_vars
