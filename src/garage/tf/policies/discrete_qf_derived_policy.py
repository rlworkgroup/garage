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
        """Vectorized or not."""
        return True

    def get_action(self, observation):
        """Get action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            Single optimal action from this policy.

        """
        q_vals = self._f_qval([observation])
        opt_action = np.argmax(q_vals)

        return opt_action

    def get_actions(self, observations):
        """Get actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment.

        Returns:
            Optimal actions from this policy.

        """
        q_vals = self._f_qval(observations)
        opt_actions = np.argmax(q_vals, axis=1)

        return opt_actions

    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = self.__dict__.copy()
        del new_dict['_f_qval']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        self.__dict__.update(state)
        self._initialize()
