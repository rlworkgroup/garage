"""EnvSpec class."""


class EnvSpec:
    """EnvSpec class.

    Args:
        observation_space (akro.Space): The observation space of the env.
        action_space (akro.Space): The action space of the env.

    """

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
