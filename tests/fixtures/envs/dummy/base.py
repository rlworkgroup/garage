"""Dummy environment for testing purpose."""
import gym


class DummyEnv(gym.Env):
    """Base dummy environment.

    Args:
        random (bool): If observations are randomly generated or not.
        obs_dim (iterable): Observation space dimension.
        action_dim (iterable): Action space dimension.

    """

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
        """Step the environment.

        Args:
            action (int): Action input.

        """
        raise NotImplementedError

    def render(self, mode='human'):
        """Render.

        Args:
            mode (str): Render mode.

        """
