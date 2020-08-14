"""Simple QFunction for testing."""
import tensorflow as tf

from garage.tf.q_functions import QFunction
from tests.fixtures.models import SimpleMLPModel


class SimpleQFunction(QFunction):
    """Simple QFunction for testing.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        name (str): Name of the q-function, also serves as the variable scope.

    """

    def __init__(self, env_spec, name='SimpleQFunction'):
        super().__init__(name)
        # avnish
        # self.obs_dim = env_spec.observation_space.shape
        self.obs_dim = (env_spec.observation_space.flat_dim, )
        action_dim = env_spec.observation_space.flat_dim
        self.model = SimpleMLPModel(output_dim=action_dim)

        self._q_val = None

        self._initialize()

    def _initialize(self):
        """Initialize QFunction."""
        obs_ph = tf.compat.v1.placeholder(tf.float32, (None, ) + self.obs_dim,
                                          name='obs')

        with tf.compat.v1.variable_scope(self.name, reuse=False) as vs:
            self._variable_scope = vs
            self._q_val = self.model.build(obs_ph).outputs

    @property
    def q_vals(self):
        """Return the Q values, the output of the network.

        Return:
            list[tf.Tensor]: Q values.

        """
        return self._q_val

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        self.__dict__.update(state)
        self._initialize()

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = self.__dict__.copy()
        del new_dict['_q_val']
        return new_dict
