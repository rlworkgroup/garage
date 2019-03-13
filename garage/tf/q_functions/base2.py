"""Q-function base classes without Parameterized."""
import tensorflow as tf


class QFunction2:
    """
    Q-function base class without Parameterzied.

    Args:
        name: Name of the Q-fucntion.

    """

    def __init__(self, name):
        self.name = name
        self._variable_scope = tf.VariableScope(reuse=False, name=name)

    def get_qval_sym(self, *input_phs):
        """
        Symbolic graph for q-network.

        All derived classes should implement this function.

        Args:
            input_phs: List of tf.Tensor inputs. recommended to be positional
            arguments, e.g. def get_qval_sym(self, state_input, action_input).
        """
        raise NotImplementedError

    def clone(self, name):
        """
        Return a clone of the Q-function.

        Args:
            name: Name of the newly created q-function.
        """
        raise NotImplementedError

    def get_trainable_vars(self):
        """Get all trainable variables under the QFunction scope."""
        return self._variable_scope.trainable_variables()

    def get_global_vars(self):
        """Get all global variables under the QFunction scope."""
        return self._variable_scope.global_variables()

    def get_regularizable_vars(self):
        """Get all regularizable variables under the QFunction scope."""
        reg_vars = [
            var for var in self.get_trainable_vars()
            if 'W' in var.name and 'output' not in var.name
        ]
        return reg_vars
