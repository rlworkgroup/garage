"""A Discrete QFunction-derived policy.

This policy chooses the action that yields to the largest Q-value.
"""
import akro
import numpy as np
import tensorflow as tf

from garage.tf.policies import Policy


class DiscreteQfDerivedPolicy(Policy):
    """DiscreteQfDerived policy.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        qf (garage.q_functions.QFunction): The q-function used.
        name (str): Name of the policy.

    """

    def __init__(self, env_spec, qf, name='DiscreteQfDerivedPolicy'):
        super().__init__(name, env_spec)

        assert isinstance(env_spec.action_space, akro.Discrete)
        self._env_spec = env_spec
        self._qf = qf

        self._initialize()

    def _initialize(self):
        self._f_qval = tf.compat.v1.get_default_session().make_callable(
            self._qf.q_vals,
            feed_list=[self._qf.model.networks['default'].input])

    @property
    def vectorized(self):
        """Vectorized or not.

        Returns:
            bool: True if vectorized.

        """
        return True

    def get_action(self, observation):
        """Get action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            int: Single optimal action from this policy.

        """
        q_vals = self._f_qval([observation])
        opt_action = np.argmax(q_vals)

        return opt_action

    def get_actions(self, observations):
        """Get actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment.

        Returns:
            numpy.ndarray: Optimal actions from this policy.

        """
        q_vals = self._f_qval(observations)
        opt_actions = np.argmax(q_vals, axis=1)

        return opt_actions

    def get_trainable_vars(self):
        """Get trainable variables.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
                variable scope.

        """
        return self._qf.get_trainable_vars()

    def get_global_vars(self):
        """Get global variables.

        Returns:
            List[tf.Variable]: A list of global variables in the current
                variable scope.

        """
        return self._qf.get_global_vars()

    def get_regularizable_vars(self):
        """Get all network weight variables in the current scope.

        Returns:
            List[tf.Variable]: A list of network weight variables in the
                current variable scope.

        """
        return self._qf.get_regularizable_vars()

    def get_params(self, trainable=True):
        """Get the trainable variables.

        Args:
            trainable (bool): Trainable or not.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
                variable scope.

        """
        return self._qf.get_params()

    def get_param_shapes(self, **tags):
        """Get parameter shapes.

        Args:
            tags: Extra arguments.

        Returns:
            List[tuple]: A list of variable shapes.

        """
        return self._qf.get_param_shapes()

    def get_param_values(self, **tags):
        """Get param values.

        Args:
            tags: Extra arguments.

        Returns:
            np.ndarray: Values of the parameters evaluated in
                the current session

        """
        return self._qf.get_param_values()

    def set_param_values(self, param_values, name=None, **tags):
        """Set param values.

        Args:
            param_values (np.ndarray): A numpy array of parameter values.
            name (str): Name of the scope.
            tags: Extra arguments.

        """
        self._qf.set_param_values(param_values)

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = self.__dict__.copy()
        del new_dict['_f_qval']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        self.__dict__.update(state)
        self._initialize()
