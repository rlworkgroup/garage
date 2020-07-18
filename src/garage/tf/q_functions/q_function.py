"""Q-function base classes without Parameterized."""
import abc


class QFunction(abc.ABC):
    """Q-function base class without Parameterzied."""

    @abc.abstractmethod
    def get_qval_sym(self, *input_phs):
        """Symbolic graph for q-network.

        All derived classes should implement this function.

        Args:
            input_phs (list[tf.Tensor]): Recommended to be positional
                arguments, e.g. def get_qval_sym(self, state_input,
                action_input).

        Return:
            tf.Tensor: The tf.Tensor output of the QFunction.

        """

    @abc.abstractmethod
    def clone(self, name):
        """Return a clone of the Q-function.

        It should only copy the configuration of the Q-function,
        not the parameters.

        Args:
            name (str): Name of the newly created q-function.
        """
