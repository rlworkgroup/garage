"""DeterministicMLPPolict with model."""
from akro.tf import Box
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.models.discrete_mlp_model import DiscreteMLPModel
from garage.tf.policies.base2 import Policy2


class DeterministicMLPPolicyWithModel(Policy2):
    """
    DeterministicMLPPolicy with model.

    It only works with Box environment.

    Args:
        env_spec: Environment specification.
        name: variable scope of the mlp.
        hidden_sizes: Output dimension of dense layer(s).
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
        layer_normalization: Bool for using layer normalization or not.

    """

    def __init__(self,
                 env_spec,
                 name="DeterministicMLPPolicy",
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=tf.nn.tanh,
                 layer_normalization=False):
        assert isinstance(
            env_spec.action_space,
            Box), ("DeterministicMLPPolicy only works with Box environment.")
        super().__init__(name, env_spec)
        self.obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        self.model = DiscreteMLPModel(
            output_dim=action_dim,
            name=name,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
            layer_normalization=layer_normalization)

        self._initialize()

    def _initialize(self):
        state_input = tf.placeholder(tf.float32, shape=(None, self.obs_dim))

        self.model.build(state_input)

        self._f_prob = tf.get_default_session().make_callable(
            self.model.networks['default'].output,
            feed_list=[self.model.networks['default'].input])

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True

    def get_action_sym(self, obs_var, name):
        """Symbolic graph to the network."""
        return self.model.build(obs_var, name=name)

    def _f_prob(self, observations):
        prob = tf.get_default_session().run(
            self.model.networks['default'].output,
            feed_dict={self.model.networks['default'].input: observations})

        return prob

    @overrides
    def get_action(self, observation):
        """Get action from the policy."""
        flat_obs = self.observation_space.flatten(observation)
        action = self._f_prob([flat_obs])[0]
        return action, dict()

    @overrides
    def get_actions(self, observations):
        """Get actions from the policy."""
        flat_obs = self.observation_space.flatten_n(observations)
        actions = self._f_prob(flat_obs)
        return actions, dict()

    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = self.__dict__.copy()
        del new_dict['_f_prob']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        self.__dict__.update(state)
        self._initialize()
