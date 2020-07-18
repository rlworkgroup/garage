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
        assert isinstance(env_spec.action_space, akro.Discrete)
        super().__init__(name)
        self._env_spec = env_spec
        self._qf = qf

        self._initialize()

    def _initialize(self):
        with tf.compat.v1.variable_scope(self.name, reuse=False) as vs:
            self._variable_scope = vs
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
