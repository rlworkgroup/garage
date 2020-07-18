"""Interface for primitives which build on top of models."""
import abc

import tensorflow as tf

from garage.misc.tensor_utils import flatten_tensors, unflatten_tensors


class Module(abc.ABC):
    """A module that builds on top of model.

    Args:
        name (str): Module name, also the variable scope.

    """

    def __init__(self, name):
        self._name = name
        self._variable_scope = None
        self._cached_params = None
        self._cached_param_shapes = None

    @property
    def name(self):
        """str: Name of this module."""
        return self._name

    def reset(self, do_resets=None):
        """Reset the module.

        This is effective only to recurrent modules. do_resets is effective
        only to vectoried modules.

        For a vectorized modules, do_resets is an array of boolean indicating
        which internal states to be reset. The length of do_resets should be
        equal to the length of inputs.

        Args:
            do_resets (numpy.ndarray): Bool array indicating which states
                to be reset.

        """

    @property
    def state_info_specs(self):
        """State info specification.

        Returns:
            List[str]: keys and shapes for the information related to the
                module's state when taking an action.

        """
        return list()

    @property
    def state_info_keys(self):
        """State info keys.

        Returns:
            List[str]: keys for the information related to the module's state
                when taking an input.

        """
        return [k for k, _ in self.state_info_specs]

    def terminate(self):
        """Clean up operation."""

    def get_trainable_vars(self):
        """Get trainable variables.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
                variable scope.

        """
        return self._variable_scope.trainable_variables()

    def get_global_vars(self):
        """Get global variables.

        Returns:
            List[tf.Variable]: A list of global variables in the current
                variable scope.

        """
        return self._variable_scope.global_variables()

    def get_regularizable_vars(self):
        """Get all network weight variables in the current scope.

        Returns:
            List[tf.Variable]: A list of network weight variables in the
                current variable scope.

        """
        trainable = self._variable_scope.global_variables()
        return [
            var for var in trainable
            if 'hidden' in var.name and 'kernel' in var.name
        ]

    def get_params(self):
        """Get the trainable variables.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
                variable scope.

        """
        if self._cached_params is None:
            self._cached_params = self.get_trainable_vars()
        return self._cached_params

    def get_param_shapes(self):
        """Get parameter shapes.

        Returns:
            List[tuple]: A list of variable shapes.

        """
        if self._cached_param_shapes is None:
            params = self.get_params()
            param_values = tf.compat.v1.get_default_session().run(params)
            self._cached_param_shapes = [val.shape for val in param_values]
        return self._cached_param_shapes

    def get_param_values(self):
        """Get param values.

        Returns:
            np.ndarray: Values of the parameters evaluated in
                the current session

        """
        params = self.get_params()
        param_values = tf.compat.v1.get_default_session().run(params)
        return flatten_tensors(param_values)

    def set_param_values(self, param_values):
        """Set param values.

        Args:
            param_values (np.ndarray): A numpy array of parameter values.

        """
        param_values = unflatten_tensors(param_values, self.get_param_shapes())
        for param, value in zip(self.get_params(), param_values):
            param.load(value)

    def flat_to_params(self, flattened_params):
        """Unflatten tensors according to their respective shapes.

        Args:
            flattened_params (np.ndarray): A numpy array of flattened params.

        Returns:
            List[np.ndarray]: A list of parameters reshaped to the
                shapes specified.

        """
        return unflatten_tensors(flattened_params, self.get_param_shapes())

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = self.__dict__.copy()
        del new_dict['_cached_params']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        self._cached_params = None
        self.__dict__.update(state)


class StochasticModule(Module):
    """Stochastic Module."""

    @property
    @abc.abstractmethod
    def distribution(self):
        """Distribution."""
