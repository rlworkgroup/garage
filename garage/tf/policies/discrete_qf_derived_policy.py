"""
Discrete QfDerived policy.

This policy chooses the action that yields to the largest q-value.
"""
from akro.tf import Discrete
import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.tf.policies.base import Policy


class DiscreteQfDerivedPolicy(Policy, Serializable):
    """
    DiscreteQfDerived policy.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        qf (garage.q_functions.QFunction): The q-function used.
    """

    def __init__(self, env_spec, qf):
        Serializable.quick_init(self, locals())
        super().__init__(env_spec)

        assert isinstance(env_spec.action_space, Discrete)
        self._env_spec = env_spec
        self._qf = qf

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True

    @overrides
    def get_action(self, observation):
        """
        Get action from this policy for the input observation.

        Args:
            observation: Observation from environment.
            sess: tf.Session provided.

        Returns:
            opt_action: Optimal action from this policy.

        """
        sess = tf.get_default_session()
        q_vals = sess.run(
            self._qf.q_val, feed_dict={self._qf.obs_ph: [observation]})
        opt_action = np.argmax(q_vals)

        return opt_action

    @overrides
    def get_actions(self, observations):
        """
        Get actions from this policy for the input observations.

        Args:
            observations: Observations from environment.
            sess: tf.Session provided.

        Returns:
            opt_actions: Optimal actions from this policy.

        """
        sess = tf.get_default_session()
        q_vals = sess.run(
            self._qf.q_val, feed_dict={self._qf.obs_ph: observations})
        opt_actions = np.argmax(q_vals, axis=1)

        return opt_actions
