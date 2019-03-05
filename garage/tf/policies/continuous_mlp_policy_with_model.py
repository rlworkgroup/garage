"""
This modules creates a continuous MLP policy network.

A continuous MLP network can be used as policy method in different RL
algorithms. It accepts an observation of the environment and predicts an
action.
"""
from akro.tf import Box
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.models.continuous_mlp_model import ContinuousMLPModel
from garage.tf.policies.base2 import Policy2


class ContinuousMLPPolicyWithModel(Policy2):
    """
    ContinuousMLPPolicy with model.

    The policy network selects action based on the state of the environment.
    It uses neural nets to fit the function of pi(s).

    Args:
        env_spec: Environment specification.
        name: variable scope of the mlp.
        hidden_sizes: Output dimension of dense layer(s).
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
        input_include_goal: Include goal or not.
        layer_normalization: Bool for using layer normalization or not.

    """

    def __init__(self,
                 env_spec,
                 name="ContinuousMLPPolicy",
                 hidden_sizes=(64, 64),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=tf.nn.tanh,
                 input_include_goal=False,
                 layer_normalization=False):
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
        super().__init__(name, env_spec)
        action_dim = env_spec.action_space.flat_dim
        if input_include_goal:
            self.obs_dim = env_spec.observation_space.flat_dim_with_keys(
                ["observation", "desired_goal"])
        else:
            self.obs_dim = env_spec.observation_space.flat_dim

        self.model = ContinuousMLPModel(
            output_dim=action_dim,
            name=name,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
            output_scale=env_spec.action_space.high,
            layer_normalization=layer_normalization)

        self._initialize()

    def _initialize(self):
        state_input = tf.placeholder(tf.float32, shape=(None, self.obs_dim))

        self.model.build(state_input)

        self._f_prob = tf.get_default_session().make_callable(
            self.model.networks['default'].output,
            feed_list=[self.model.networks['default'].input])

    def get_action_sym(self, obs_var, name=None, **kwargs):
        """Return action sym according to obs_var."""
        return self.model.build(obs_var, name=name)

    @overrides
    def get_action(self, observation):
        """Return a single action."""
        return self._f_prob([observation])[0], dict()

    @overrides
    def get_actions(self, observations):
        """Return multiple actions."""
        return self._f_prob(observations), dict()

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True

    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = self.__dict__.copy()
        del new_dict['_f_prob']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        self.__dict__.update(state)
        self._initialize()
