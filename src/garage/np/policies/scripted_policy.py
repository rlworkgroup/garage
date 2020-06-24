"""Simulates a garage policy object."""
from garage.np.policies.policy import Policy


class ScriptedPolicy(Policy):
    """Simulates a garage policy object.

    Args:
        scripted_actions (list or dictionary): data structure indexed by
            observation, returns a corresponding action
        agent_env_infos (list or dictionary): data structure indexed by
            observation, returns a corresponding agent_env_info

    """

    def __init__(self, scripted_actions, agent_env_infos=None):
        self._scripted_actions = scripted_actions
        self._agent_env_infos = agent_env_infos

    def set_param_values(self, params):
        """Set param values.

        Args:
            params (np.ndarray): A numpy array of parameter values.

        """

    # pylint: disable=no-self-use
    def get_param_values(self):
        """Get param values.

        Returns:
            np.ndarray: Values of the parameters evaluated in
                the current session

        """
        return []

    def get_action(self, observation):
        """Return a single action.

        Args:
            observation (numpy.ndarray): Observations.

        Returns:
            int: Action given input observation.
            dict[dict]: Agent infos indexed by observation.

        """
        if self._agent_env_infos:
            a_info = self._agent_env_infos[observation]
        else:
            a_info = dict()
        return self._scripted_actions[observation], a_info

    def get_actions(self, observations):
        """Return multiple actions.

        Args:
            observations (numpy.ndarray): Observations.

        Returns:
            list[int]: Actions given input observations.
            dict[dict]: Agent info indexed by observation.

        """
        if self._agent_env_infos:
            a_info = self._agent_env_infos[observations[0]]
        else:
            a_info = dict()
        return [self._scripted_actions[obs] for obs in observations], a_info
