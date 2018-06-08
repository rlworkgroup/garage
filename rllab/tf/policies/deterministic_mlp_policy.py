import tensorflow as tf

from rllab.core import Serializable
from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.tf.core import LayersPowered
from rllab.tf.core import MLP
from rllab.tf.distributions import Categorical
from rllab.tf.policies import Policy
from rllab.tf.misc import tensor_utils

import rllab.tf.core.layers as L
from rllab.tf.core.layers import batch_norm

from rllab.tf.spaces import Discrete
import tensorflow as tf


class DeterministicMLPPolicy(Policy, LayersPowered, Serializable):
    def __init__(self,
                 env_spec,
                 name="DeterministicMLPPolicy",
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=tf.nn.tanh,
                 prob_network=None,
                 bn=False):
        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):
            if prob_network is None:
                prob_network = MLP(
                    input_shape=(env_spec.observation_space.flat_dim, ),
                    output_dim=env_spec.action_space.flat_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    # batch_normalization=True,
                    name="prob_network",
                )

            self._l_prob = prob_network.output_layer
            self._l_obs = prob_network.input_layer
            self._f_prob = tensor_utils.compile_function(
                [prob_network.input_layer.input_var],
                L.get_output(prob_network.output_layer, deterministic=True))

        self.prob_network = prob_network

        # Note the deterministic=True argument. It makes sure that when getting
        # actions from single observations, we do not update params in the
        # batch normalization layers.
        # TODO: this doesn't currently work properly in the tf version so we
        # leave out batch_norm
        super(DeterministicMLPPolicy, self).__init__(env_spec)
        LayersPowered.__init__(self, [prob_network.output_layer])

    @property
    def vectorized(self):
        return True

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        action = self._f_prob([flat_obs])[0]
        return action, dict()

    @overrides
    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        actions = self._f_prob(flat_obs)
        return actions, dict()

    def get_action_sym(self, obs_var):
        return L.get_output(self.prob_network.output_layer, obs_var)
