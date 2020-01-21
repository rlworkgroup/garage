"""Policy that performs a fixed sequence of actions."""
import numpy as np

from garage.misc.tensor_utils import stack_tensor_dict_list
from garage.np.policies.base import Policy


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
        super().__init__(env_spec)
        if agent_infos is None:
            agent_infos = [{}] * len(scripted_actions)
        self._scripted_actions = scripted_actions
        self._agent_infos = agent_infos
        self._indices = [0]

    def reset(self, dones=None):
        """Reset policy.

        Args:
            dones (None or list[bool]): Vectorized policy states to reset.

        """
        if dones is None:
            dones = [True]
        while len(dones) > len(self._indices):
            self._indices.append(0)
        while len(self._indices) > len(dones):
            self._indices.pop()
        for i, d in enumerate(dones):
            if d:
                self._indices[i] = 0

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
        if len(self._indices) > 1:
            raise ValueError('Cannot call ScriptedPolicy.get_action while '
                             'vectorized.')
        action = self._scripted_actions[self._indices[0]]
        agent_info = self._agent_infos[self._indices[0]]
        return action, agent_info

    def get_actions(self, observations):
        """Get next action.

        Args:
            observations (np.ndarray): Ignored.

        Returns:
            tuple[np.ndarray, dict[str, np.ndarray]]: The action and agent_info
                for this time step.

        """
        del observations
        actions = []
        agent_infos = []
        for index in self._indices:
            actions.append(self._scripted_actions[self._indices[index]])
            agent_infos.append(self._agent_infos[self._indices[index]])
        return np.asarray(actions), stack_tensor_dict_list(agent_infos)
