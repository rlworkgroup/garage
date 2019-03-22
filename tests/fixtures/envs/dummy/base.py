import gym


class DummyEnv(gym.Env):
    """Base dummy environment."""

    def __init__(self, random, obs_dim=(4, ), action_dim=(2, )):
        self.random = random
        self.state = None
        self._obs_dim = obs_dim
        self._action_dim = action_dim

    @property
    def observation_space(self):
        """Return an observation space."""
        raise NotImplementedError

    @property
    def action_space(self):
        """Return an action space."""
        raise NotImplementedError

    def reset(self):
        """Reset the environment."""
        raise NotImplementedError

    def step(self, action):
        """Step the environment."""
        raise NotImplementedError
