class ScriptedPolicy:
    """
    A mock policy for 4x4 gridworldenv

    '4x4': [
        'SFFF',
        'FHFH',
        'FFFH',
        'HFFG'
    ]

    0: left
    1: down
    2: right
    3: up
    -1: no move

    'S' : starting point
    'F' or '.': free space
    'W' or 'x': wall
    'H' or 'o': hole (terminates episode)
    'G' : goal
    [2,2,1,0,3,1,1,1,2,2,1,1,1,2,2,1]
    """

    def __init__(self):
        self.scriptedActions = [2, 2, 1, 0, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1]

    def set_param_values(self, params):
        return params

    def get_param_values(self):
        return 'ScriptedPolicy'

    def reset(self, dones=None):
        pass

    def get_action(self, obs):
        return self.scriptedActions[obs], {}

    def get_actions(self, obses):
        return [self.scriptedActions[obs] for obs in obses], {}
