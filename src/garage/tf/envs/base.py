"""Garage wrapper for environments used for Tensorflow."""

import akro
from cached_property import cached_property

from garage.envs import GarageEnv


class TfEnv(GarageEnv):
    """
    Returns a TensorFlow wrapper class for gym.Env.

    Args:
        env (gym.Env): the env that will be wrapped
    """

    def __init__(self, env=None, env_name=''):
        super().__init__(env, env_name)
        self.action_space = akro.from_gym(self.env.action_space)
        self.observation_space = akro.from_gym(self.env.observation_space)

    @cached_property
    def max_episode_steps(self):
        """Return gym.Env's max episode steps.

        Returns:
            max_episode_steps (int)

        """
        return self.env.spec.max_episode_steps
