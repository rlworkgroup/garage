"""Wrapper class that converts gym.Env into GarageEnv."""
import akro
import gym
import numpy as np

from garage.envs.env_spec import EnvSpec


class RL2Env(gym.Wrapper):
    """An wrapper class for gym.Env for RL^2.

    RL^2 augments the observation by concatenating the new observation
    with previous action, reward and terminal signal.

    Args:
        env (gym.Env): An env that will be wrapped.

    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = akro.from_gym(self.env.action_space)
        self.observation_space = akro.from_gym(self._create_rl2_obs_space(env))
        self.spec = EnvSpec(action_space=self.action_space,
                            observation_space=self.observation_space)

    def _create_rl2_obs_space(self, env):
        # pylint: disable=no-self-use
        """Create an augmented observation space from the input environment.

        This is specific to RL^2.

        Args:
            env (gym.Env): An env that will be used.

        Returns:
            gym.spaces.Box: The augmented observation space.

        """
        obs_flat_dim = np.prod(env.observation_space.shape)
        action_flat_dim = np.prod(env.action_space.shape)
        return gym.spaces.Box(low=env.observation_space.low[0],
                              high=env.observation_space.high[0],
                              shape=(obs_flat_dim + action_flat_dim + 1 + 1, ))

    def reset(self):
        # pylint: disable=arguments-differ
        """Reset the environment.

        Returns:
            np.ndarray: The augmented observation obtained after reset.

        """
        obs = self.env.reset()
        return np.concatenate(
            [obs, np.zeros(self.env.action_space.shape), [0], [0]])

    def step(self, action):
        """Step the environment.

        Args:
            action (int): Action input.

        Returns:
            np.ndarray: Observation.
            float: Reward.
            bool: If the environment is terminated.
            dict: Environment information.

        """
        next_obs, reward, done, info = self.env.step(action)
        next_obs = np.concatenate([next_obs, action, [reward], [done]])
        return next_obs, reward, done, info
