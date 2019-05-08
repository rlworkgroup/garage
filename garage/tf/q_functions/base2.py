"""Q-function base classes without Parameterized."""


class QFunction2:
    """
    Q-function base class without Parameterzied.

    Args:
        name (str): Name of the Q-fucntion, also the variable scope.

    """

    def __init__(self, name):
        self.name = name or type(self).__name__
        self._variable_scope = None

    def get_qval_sym(self, *input_phs):
        """
        Symbolic graph for q-network.

        All derived classes should implement this function.

        Args:
            input_phs (list[tf.Tensor]): Recommended to be positional
            arguments, e.g. def get_qval_sym(self, state_input, action_input).
        """
        raise NotImplementedError

    def clone(self, name):
        """
        Return a clone of the Q-function.

        Args:
            name (str): Name of the newly created q-function.
        """
        raise NotImplementedError

    def get_trainable_vars(self):
        """Get all trainable variables under the QFunction scope."""
        return self._variable_scope.trainable_variables()

    def get_global_vars(self):
        """Get all global variables under the QFunction scope."""
        return self._variable_scope.global_variables()
