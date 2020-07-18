"""Simple QFunction for testing."""
import tensorflow as tf

from garage.tf.q_functions import QFunction

from tests.fixtures.models import SimpleMLPModel


class SimpleQFunction(SimpleMLPModel, QFunction):
    """Simple QFunction for testing.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        name (str): Name of the q-function, also serves as the variable scope.

    """

    def __init__(self, env_spec, name='SimpleQFunction'):
        self.obs_dim = env_spec.observation_space.shape
        action_dim = env_spec.observation_space.flat_dim
        super().__init__(output_dim=action_dim, name=name)

        self._q_val = None

        self._initialize()

    def _initialize(self):
        """Initialize QFunction."""
        obs_ph = tf.compat.v1.placeholder(tf.float32, (None, ) + self.obs_dim,
                                          name='obs')

        self._q_val = super().build(obs_ph).outputs

    @property
    def q_vals(self):
        """Return the Q values, the output of the network.

        Return:
            list[tf.Tensor]: Q values.

        """
        return self._q_val

    def get_qval_sym(self, *input_phs):
        """Intantiate abstract method.

        Args:
            input_phs (list[tf.Tensor]): Recommended to be positional
                arguments, e.g. def get_qval_sym(self, state_input,
                action_input).
        """

    def clone(self, name):
        """Intantiate abstract method.

        Args:
            name (str): Name of the newly created q-function.
        """

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_q_val']
        return new_dict
