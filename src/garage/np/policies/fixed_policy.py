"""Policy that performs a fixed sequence of actions."""
import numpy as np

from garage.np.policies.policy import Policy


class FixedPolicy(Policy):
    """Policy that performs a fixed sequence of actions.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        scripted_actions (list[np.ndarray] or np.ndarray): Sequence of actions
            to perform.
        agent_infos (list[dict[str, np.ndarray]] or None): Sequence of
            agent_infos to produce.

    """

    def __init__(self, env_spec, scripted_actions, agent_infos=None):
        if agent_infos is None:
            agent_infos = [{}] * len(scripted_actions)
        self._env_spec = env_spec
        self._scripted_actions = scripted_actions
        self._agent_infos = agent_infos
        self._indices = [0]

    def reset(self, do_resets=None):
        """Reset policy.

        Args:
            do_resets (None or list[bool]): Vectorized policy states to reset.

        Raises:
            ValueError: If do_resets has length greater than 1.

        """
        if do_resets is None:
            do_resets = [True]
        if len(do_resets) > 1:
            raise ValueError('FixedPolicy does not support more than one '
                             'action at a time.')
        self._indices[0] = 0

    def set_param_values(self, params):
        """Set param values of policy.

        Args:
            params (object): Ignored.

        """
        # pylint: disable=no-self-use
        del params

    def get_param_values(self):
        """Return policy params (there are none).

        Returns:
            tuple: Empty tuple.

        """
        # pylint: disable=no-self-use
        return ()

    def get_action(self, observation):
        """Get next action.

        Args:
            observation (np.ndarray): Ignored.

        Raises:
            ValueError: If policy is currently vectorized (reset was called
                with more than one done value).

        Returns:
            tuple[np.ndarray, dict[str, np.ndarray]]: The action and agent_info
                for this time step.

        """
        del observation
        action = self._scripted_actions[self._indices[0]]
        agent_info = self._agent_infos[self._indices[0]]
        self._indices[0] += 1
        return action, agent_info

    def get_actions(self, observations):
        """Get next action.

        Args:
            observations (np.ndarray): Ignored.

        Raises:
            ValueError: If observations has length greater than 1.

        Returns:
            tuple[np.ndarray, dict[str, np.ndarray]]: The action and agent_info
                for this time step.

        """
        if len(observations) != 1:
            raise ValueError('FixedPolicy does not support more than one '
                             'observation at a time.')
        action, agent_info = self.get_action(observations[0])
        return np.array(
            [action]), {k: np.array([v])
                        for (k, v) in agent_info.items()}

    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """
        return self._env_spec
