import tensorflow as tf

from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.tf.core import LayersPowered
import garage.tf.core.layers as L
from garage.tf.core.network import MLP
from garage.tf.misc import tensor_utils
from garage.tf.policies import Policy
from garage.tf.spaces import Box


class DeterministicMLPPolicy(Policy, LayersPowered, Serializable):
    def __init__(self,
                 env_spec,
                 name="DeterministicMLPPolicy",
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=tf.nn.tanh,
                 prob_network=None,
                 bn=False):
        assert isinstance(env_spec.action_space, Box)

        Serializable.quick_init(self, locals())

        self._prob_network_name = "prob_network"
        with tf.variable_scope(name, "DeterministicMLPPolicy"):
            if prob_network is None:
                prob_network = MLP(
                    input_shape=(env_spec.observation_space.flat_dim, ),
                    output_dim=env_spec.action_space.flat_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    # batch_normalization=True,
                    name="mlp_prob_network",
                )

            with tf.name_scope(self._prob_network_name):
                prob_network_output = L.get_output(
                    prob_network.output_layer, deterministic=True)
            self._l_prob = prob_network.output_layer
            self._l_obs = prob_network.input_layer
            self._f_prob = tensor_utils.compile_function(
                [prob_network.input_layer.input_var], prob_network_output)

        self.prob_network = prob_network
        self.name = name

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

    def get_action_sym(self, obs_var, name=None):
        with tf.name_scope(name, "get_action_sym", values=[obs_var]):
            with tf.name_scope(self._prob_network_name, values=[obs_var]):
                return L.get_output(self.prob_network.output_layer, obs_var)
