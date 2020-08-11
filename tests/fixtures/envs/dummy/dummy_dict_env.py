"""Dummy akro.Dict environment for testing purpose."""
import akro
import gym
import numpy as np

from garage import EnvSpec

from tests.fixtures.envs.dummy import DummyEnv


class DummyDictEnv(DummyEnv):
    """A dummy akro.Dict environment with predefined inner spaces.

    Args:
        random (bool): If observations are randomly generated or not.
        obs_space_type (str): The type of the inner spaces of the
            dict observation space.
        act_space_type (str): The type of action space to mock.

    """

    def __init__(self,
                 random=True,
                 obs_space_type='box',
                 act_space_type='box'):
        assert obs_space_type in ['box', 'image', 'discrete']
        assert act_space_type in ['box', 'discrete']
        super().__init__(random)
        self.obs_space_type = obs_space_type
        self.act_space_type = act_space_type
        self.spec = EnvSpec(action_space=self.action_space,
                            observation_space=self.observation_space)

    @property
    def observation_space(self):
        """Return the observation space.

        Returns:
            akro.Dict: Observation space.
        """
        if self.obs_space_type == 'box':
            return gym.spaces.Dict({
                'achieved_goal':
                gym.spaces.Box(low=-200.,
                               high=200.,
                               shape=(3, ),
                               dtype=np.float32),
                'desired_goal':
                gym.spaces.Box(low=-200.,
                               high=200.,
                               shape=(3, ),
                               dtype=np.float32),
                'observation':
                gym.spaces.Box(low=-200.,
                               high=200.,
                               shape=(25, ),
                               dtype=np.float32)
            })
        elif self.obs_space_type == 'image':
            return gym.spaces.Dict({
                'dummy':
                gym.spaces.Box(low=0,
                               high=255,
                               shape=(100, 100, 3),
                               dtype=np.uint8),
            })
        else:
            return gym.spaces.Dict({'dummy': gym.spaces.Discrete(5)})

    @property
    def action_space(self):
        """Return the action space.

        Returns:
            akro.Box: Action space.

        """
        if self.act_space_type == 'box':
            return akro.Box(low=-5.0, high=5.0, shape=(1, ), dtype=np.float32)
        else:
            return akro.Discrete(5)

    def reset(self):
        """Reset the environment.

        Returns:
            numpy.ndarray: Observation after reset.

        """
        return self.observation_space.sample()

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
        return self.observation_space.sample(), 0, True, dict()

    # pylint: disable=no-self-use
    def compute_reward(self, achieved_goal, goal, info):
        """Function to compute new reward.

        Args:
            achieved_goal (numpy.ndarray): Achieved goal.
            goal (numpy.ndarray): Original desired goal.
            info (dict): Extra information.

        Returns:
            float: New computed reward.

        """
        del info
        return np.sum(achieved_goal - goal)
