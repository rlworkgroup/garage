"""
QfDerived policy.

This policy chooses the action that yields to the largest q-value.
"""
import numpy as np

from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.spaces import Discrete
from garage.tf.policies import Policy


class QfDerivedPolicy(Policy, Serializable):
    """
    QfDerived policy.

    Args:
        env_spec: Environment specification.
        qf: The q-function used.
        obs_ph: The place holder for observation.
    """

    def __init__(self, env_spec, qf, obs_ph):
        Serializable.quick_init(self, locals())
        super().__init__(env_spec)

        assert isinstance(env_spec.action_space, Discrete)
        self._env_spec = env_spec
        self._qf = qf
        self._obs_ph = obs_ph

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True

    @overrides
    def get_action(self, observation, sess=None):
        """
        Get action from this policy for the input observation.

        Args:
            observation: Observation from environment.

        Returns:
            opt_action: Optimal action from this policy.

        """
        q_vals = sess.run(self._qf, feed_dict={self._obs_ph: [observation]})
        opt_action = np.argmax(q_vals)

        return opt_action

    @overrides
    def get_actions(self, observations, sess=None):
        """
        Get actions from this policy for the input observations.

        Args:
            observations: Observations from environment.

        Returns:
            opt_actions: Optimal actions from this policy.

        """
        q_vals = sess.run(self._qf, feed_dict={self._obs_ph: observations})
        opt_actions = np.argmax(q_vals, axis=1)

        return opt_actions
