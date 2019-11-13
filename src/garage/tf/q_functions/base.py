"""Q-function base classes without Parameterized."""
import abc


class QFunction(abc.ABC):
    """Q-function base class without Parameterzied.

    Args:
        name (str): Name of the Q-fucntion, also the variable scope.

    """

    def __init__(self, name):
        self.name = name or type(self).__name__
        self._variable_scope = None

    def get_qval_sym(self, *input_phs):
        """Symbolic graph for q-network.

        All derived classes should implement this function.

        Args:
            input_phs (list[tf.Tensor]): Recommended to be positional
                arguments, e.g. def get_qval_sym(self, state_input,
                action_input).
        """

    def clone(self, name):
        """Return a clone of the Q-function.

        It should only copy the configuration of the Q-function,
        not the parameters.

        Args:
            name (str): Name of the newly created q-function.
        """

    def get_trainable_vars(self):
        """Get all trainable variables under the QFunction scope."""
        return self._variable_scope.trainable_variables()

    def get_global_vars(self):
        """Get all global variables under the QFunction scope."""
        return self._variable_scope.global_variables()

    def get_regularizable_vars(self):
        """Get all network weight variables under the QFunction scope."""
        trainable = self._variable_scope.global_variables()
        return [
            var for var in trainable
            if 'hidden' in var.name and 'kernel' in var.name
        ]

    def log_diagnostics(self, paths):
        """Log extra information per iteration based on the collected paths."""
