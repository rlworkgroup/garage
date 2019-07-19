class ScriptedPolicy:
    """Simulates a garage policy object.

    Args:
        - scripted actions(list or dictionary): data structure indexed by
            obervation, returns a corresponding action
        - agent_env_infos(list or dictionary): data structure indexed by
            obervation, returns a corresponding agent_env_info
    """

    def __init__(self, scripted_actions, agent_env_infos=None):
        self.scripted_actions = scripted_actions
        self.agent_env_infos = agent_env_infos

    def set_param_values(self, params):
        return params

    def get_param_values(self):
        return 'ScriptedPolicy'

    def reset(self, dones=None):
        pass

    def get_action(self, obs):
        if self.agent_env_infos:
            a_info = self.agent_env_infos[obs]
        else:
            a_info = dict()
        return self.scripted_actions[obs], a_info

    def get_actions(self, obses):
        if self.agent_env_infos:
            a_info = self.agent_env_infos[obses[0]]
        else:
            a_info = dict()
        return [self.scripted_actions[obs] for obs in obses], a_info
