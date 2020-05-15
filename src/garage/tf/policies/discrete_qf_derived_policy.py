"""A Discrete QFunction-derived policy.

This policy chooses the action that yields to the largest Q-value.
"""
import akro
import numpy as np
import tensorflow as tf

from garage.tf.policies.policy import Policy


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
            self._qf.q_vals, feed_list=[self._qf.model.input])

    @property
    def vectorized(self):
        """Vectorized or not.

        Returns:
            Bool: True if primitive supports vectorized operations.

        """
        return True

    def get_action(self, observation):
        """Get action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            numpy.ndarray: Single optimal action from this policy.
            dict: Predicted action and agent information. It returns an empty
                dict since there is no parameterization.

        """
        if isinstance(self.env_spec.observation_space, akro.Image) and \
                len(observation.shape) < \
                len(self.env_spec.observation_space.shape):
            observation = self.env_spec.observation_space.unflatten(
                observation)
        q_vals = self._f_qval([observation])
        opt_action = np.argmax(q_vals)

        return opt_action, dict()

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
