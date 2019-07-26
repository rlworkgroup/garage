"""Simulates a garage policy object."""


class ScriptedPolicy:
    """Simulates a garage policy object.

    Args:
        - scripted actions(list or dictionary): data structure indexed by
            obervation, returns a corresponding action
        - agent_env_infos(list or dictionary): data structure indexed by
            obervation, returns a corresponding agent_env_info
    """

    def __init__(self, scripted_actions, agent_env_infos=None):
        self._scripted_actions = scripted_actions
        self._agent_env_infos = agent_env_infos

    def set_param_values(self, params):
        """Set param values of policy."""
        pass

    def get_param_values(self):
        """Return policy params as a list."""
        return []

    def reset(self, dones=None):
        """Reset Policy to initial state."""
        pass

    def get_action(self, obs):
        """Return action sampled from the policy."""
        if self._agent_env_infos:
            a_info = self._agent_env_infos[obs]
        else:
            a_info = dict()
        return self._scripted_actions[obs], a_info

    def get_actions(self, obses):
        """Return ACTIONS sampled from the policy."""
        if self._agent_env_infos:
            a_info = self._agent_env_infos[obses[0]]
        else:
            a_info = dict()
        return [self._scripted_actions[obs] for obs in obses], a_info
