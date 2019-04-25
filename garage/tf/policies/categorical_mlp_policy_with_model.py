"""CategoricalMLPPolicy with model."""
from akro.tf import Discrete
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.distributions import Categorical
from garage.tf.models import MLPModel
from garage.tf.policies.base2 import StochasticPolicy2


class CategoricalMLPPolicyWithModel(StochasticPolicy2):
    """
    CategoricalMLPPolicy with model.

    A policy that contains a MLP to make prediction based on
    a categorical distribution.

    It only works with akro.tf.Discrete action space.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        name (str): Policy name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means the MLP of this policy consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 env_spec,
                 name="CategoricalMLPPolicy",
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 output_nonlinearity=tf.nn.softmax,
                 layer_normalization=False):
        assert isinstance(
            env_spec.action_space,
            Discrete), ("CategoricalMLPPolicy only works with akro.tf.Discrete"
                        "action space.")
        super().__init__(name, env_spec)
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.n

        self.model = MLPModel(
            output_dim=self.action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
            layer_normalization=layer_normalization,
            name='MLPModel')

        self._initialize()

    def _initialize(self):
        state_input = tf.placeholder(tf.float32, shape=(None, self.obs_dim))

        with tf.variable_scope(self._variable_scope):
            self.model.build(state_input)

        self._f_prob = tf.get_default_session().make_callable(
            self.model.networks['default'].outputs,
            feed_list=[self.model.networks['default'].input])

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None, name=None):
        """Symbolic graph of the distribution."""
        with tf.variable_scope(self._variable_scope):
            prob = self.model.build(obs_var, name=name)
        return dict(prob=prob)

    @overrides
    def dist_info(self, obs, state_infos=None):
        """Distribution info."""
        prob = self._f_prob(obs)
        return dict(prob=prob)

    @overrides
    def get_action(self, observation):
        """Return a single action."""
        flat_obs = self.observation_space.flatten(observation)
        prob = self._f_prob([flat_obs])[0]
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        """Return multiple actions."""
        flat_obs = self.observation_space.flatten_n(observations)
        probs = self._f_prob(flat_obs)
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

    @property
    def distribution(self):
        """Policy distribution."""
        return Categorical(self.action_dim)

    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = self.__dict__.copy()
        del new_dict['_f_prob']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        self.__dict__.update(state)
        self._initialize()
