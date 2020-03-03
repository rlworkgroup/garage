"""Wrapper class that is specifc for RL2."""
import akro
import gym
import numpy as np

from garage.envs.env_spec import EnvSpec


class RL2Env(gym.Wrapper):
    """Environment wrapper for RL2.

    In RL2, observation is concatenated with previous action,
    reward and terminal signal to form new observation.

    Also, different tasks could have different observation dimension.
    An example is in ML45 from MetaWorld (reference:
    https://arxiv.org/pdf/1910.10897.pdf). This wrapper pads the
    observation to the maximum observation dimension with zeros.

    Args:
        env (gym.Env): An env that will be wrapped.
        max_obs_dim (int): Maximum observation dimension in the environments
             or tasks. Set to None when it is not applicable.

    """

    def __init__(self, env, max_obs_dim=None):
        super().__init__(env)
        self._max_obs_dim = max_obs_dim
        action_space = akro.from_gym(self.env.action_space)
        observation_space = self._create_rl2_obs_space(env)
        self._spec = EnvSpec(action_space=action_space,
                             observation_space=observation_space)

    def _create_rl2_obs_space(self, env):
        """Create observation space for RL2.

        Args:
            env (gym.Env): An env that will be wrapped.

        Returns:
            gym.spaces.Box: Augmented observation space.

        """
        obs_flat_dim = np.prod(env.observation_space.shape)
        action_flat_dim = np.prod(env.action_space.shape)
        if self._max_obs_dim is not None:
            obs_flat_dim = self._max_obs_dim
        return akro.Box(low=-np.inf,
                        high=np.inf,
                        shape=(obs_flat_dim + action_flat_dim + 1 + 1, ))

    # pylint: disable=arguments-differ
    def reset(self):
        """gym.Env reset function.

        Returns:
            np.ndarray: augmented observation.

        """
        obs = self.env.reset()
        # pad zeros if needed for running ML45
        if self._max_obs_dim is not None:
            obs = np.concatenate(
                [obs, np.zeros(self._max_obs_dim - obs.shape[0])])
        return np.concatenate(
            [obs, np.zeros(self.env.action_space.shape), [0], [0]])

    def step(self, action):
        """gym.Env step function.

        Args:
            action (int): action taken.

        Returns:
            np.ndarray: augmented observation.
            float: reward.
            bool: terminal signal.
            dict: environment info.

        """
        next_obs, reward, done, info = self.env.step(action)
        if self._max_obs_dim is not None:
            next_obs = np.concatenate(
                [next_obs,
                 np.zeros(self._max_obs_dim - next_obs.shape[0])])
        next_obs = np.concatenate([next_obs, action, [reward], [done]])
        return next_obs, reward, done, info

    @property
    def spec(self):
        """Environment specification.

        Returns:
            EnvSpec: Environment specification.

        """
        return self._spec
