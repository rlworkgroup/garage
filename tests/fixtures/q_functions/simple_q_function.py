import tensorflow as tf

from garage.tf.q_functions import QFunction2
from tests.fixtures.models import SimpleMLPModel


class SimpleQFunction(QFunction2):
    """Simple QFunction for testing."""

    def __init__(self, env_spec, name='SimpleQFunction'):
        super().__init__(name)
        self.obs_dim = env_spec.observation_space.shape
        action_dim = env_spec.observation_space.flat_dim
        self.model = SimpleMLPModel(output_dim=action_dim)

        self._initialize()

    def _initialize(self):
        obs_ph = tf.placeholder(
            tf.float32, (None, ) + self.obs_dim, name='obs')

        with tf.variable_scope(self.name, reuse=False) as self._variable_scope:
            self.model.build(obs_ph)

    @property
    def q_vals(self):
        return self.model.networks['default'].outputs

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._initialize()
