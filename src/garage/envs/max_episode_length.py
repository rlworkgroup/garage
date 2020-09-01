"""Wrapper which changes the max_episode_length."""
from garage import EnvSpec, Wrapper


class MaxEpisodeLength(Wrapper):
    """Changes the `max_episode_length` of the wrapped environment.

    Args:
        env (Environment): :class:`Environment` to wrap
        max_episode_length (int): New maximum episode length
    """

    def __init__(self, env, max_episode_length):
        super().__init__(env)
        self._spec = EnvSpec(self._env.spec.observation_space,
                             self._env.spec.action_space, max_episode_length)

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._spec
