"""A Discrete QFunction-derived policy.

This policy chooses the action that yields to the largest Q-value.
"""
import akro
import numpy as np
import tensorflow as tf

from garage.tf.models import Module
from garage.tf.policies.policy import Policy


class DiscreteQfDerivedPolicy(Module, Policy):
    """DiscreteQfDerived policy.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        qf (garage.q_functions.QFunction): The q-function used.
        name (str): Name of the policy.

    """

    def __init__(self, env_spec, qf, name='DiscreteQfDerivedPolicy'):
        assert isinstance(env_spec.action_space, akro.Discrete), (
            'DiscreteQfDerivedPolicy only supports akro.Discrete action spaces'
        )
        if isinstance(env_spec.observation_space, akro.Dict):
            raise ValueError('CNN policies do not support'
                             'with akro.Dict observation spaces.')
        super().__init__(name)
        self._env_spec = env_spec
        self._qf = qf

        self._initialize()

    def _initialize(self):
        # pylint: disable=protected-access
        self._f_qval = tf.compat.v1.get_default_session().make_callable(
            self._qf.q_vals, feed_list=[self._qf.input])

    def get_action(self, observation):
        """Get action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            numpy.ndarray: Single optimal action from this policy.
            dict: Predicted action and agent information. It returns an empty
                dict since there is no parameterization.

        """
        opt_actions, agent_infos = self.get_actions([observation])
        return opt_actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        """Get actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment.

        Returns:
            numpy.ndarray: Optimal actions from this policy.
            dict: Predicted action and agent information. It returns an empty
                dict since there is no parameterization.

        """
        if isinstance(self.env_spec.observation_space, akro.Image) and \
                len(observations[0].shape) < \
                len(self.env_spec.observation_space.shape):
            observations = self.env_spec.observation_space.unflatten_n(
                observations)
        q_vals = self._f_qval(observations)
        opt_actions = np.argmax(q_vals, axis=1)

        return opt_actions, dict()

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

    def get_params(self):
        """Get the trainable variables.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
                variable scope.

        """
        return self._qf.get_params()

    def get_param_shapes(self):
        """Get parameter shapes.

        Returns:
            List[tuple]: A list of variable shapes.

        """
        return self._qf.get_param_shapes()

    def get_param_values(self):
        """Get param values.

        Returns:
            np.ndarray: Values of the parameters evaluated in
                the current session

        """
        return self._qf.get_param_values()

    def set_param_values(self, param_values):
        """Set param values.

        Args:
            param_values (np.ndarray): A numpy array of parameter values.

        """
        self._qf.set_param_values(param_values)

    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """
        return self._env_spec

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_qval']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
