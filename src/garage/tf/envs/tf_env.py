"""Garage wrapper for environments used for Tensorflow."""

import akro
from cached_property import cached_property

from garage.envs import GarageEnv


class TfEnv(GarageEnv):
    """Returns a TensorFlow wrapper class for gym.Env.

    Args:
        env (gym.Env): the env that will be wrapped
        env_name (string): If the env_name is speficied, a gym environment
            with that name will be created. If such an environment does not
            exist, a `gym.error` is thrown.
        is_image (bool): True if observations contain pixel values,
            false otherwise. Setting this to true converts a gym.Spaces.Box
            obs space to an akro.Image and normalizes pixel values.

    """

    def __init__(self, env=None, env_name='', is_image=False):
        super().__init__(env, env_name, is_image=is_image)
        self.action_space = akro.from_gym(self.env.action_space)
        self.observation_space = akro.from_gym(self.env.observation_space,
                                               is_image=is_image)

    @cached_property
    def max_episode_steps(self):
        """Return gym.Env's max episode steps.

        Returns:
            int: Max episode steps.

        """
        return self.env.spec.max_episode_steps
