"""A dummy discrete pixel env."""
from unittest import mock

import akro
import numpy as np

from garage import EnvSpec

from tests.fixtures.envs.dummy import DummyEnv


class DummyDiscretePixelEnv(DummyEnv):
    """A dummy discrete pixel environment.

    It follows Atari game convention, where actions are 'NOOP', 'FIRE', ...
    It also contains self.unwrapped.ale.lives, get_action_meanings for testing.

    Several properties are made for testing purpose as following:

    -Observations are
        after reset    : np.ones(self._shape).
        action 1 (FIRE): np.full(self._shape, 2).
        otherwise      : random if self.random is True,
            otherwise previous state + action.

    -The environment has 5 lives.
    -Done will be True if
        -all 5 lives are exhausted
        -env.step(2), followed by env.step(1)

    Args:
        random (bool): Whether observations are generated randomly.

    """

    def __init__(self, random=True):
        super().__init__(random, obs_dim=(100, 100, 3), action_dim=5)
        self.unwrapped.get_action_meanings = self._get_action_meanings
        self.unwrapped.ale = mock.Mock()
        self.unwrapped.ale.lives = self.get_lives
        self._observation_space = akro.Box(low=0,
                                           high=255,
                                           shape=self._obs_dim,
                                           dtype=np.uint8)
        self.step_called = 0
        self._lives = None
        self._prev_action = None

    @property
    def observation_space(self):
        """akro.Box: Observation space of this environment."""
        return akro.from_gym(self._observation_space)

    @observation_space.setter
    def observation_space(self, observation_space):
        """Observation space setter.

        Args:
            observation_space (akro.Box): Observation space to be set.

        """
        self._observation_space = observation_space

    @property
    def action_space(self):
        """akro.Discrete: an action space."""
        return akro.Discrete(self._action_dim)

    @property
    def spec(self):
        """EnvSpec: the environment specification."""
        return EnvSpec(observation_space=self.observation_space,
                       action_space=self.action_space)

    # pylint: disable=no-self-use
    def _get_action_meanings(self):
        """Action meanings.

        Returns:
            list[str]: Meaning of action, indices are aligned with actions.

        """
        return ['NOOP', 'FIRE', 'SLEEP', 'EAT', 'PLAY']

    def get_lives(self):
        """Get number of lives.

        Returns:
            int: Number of lives remaining.

        """
        return self._lives

    def reset(self):
        """Reset the environment.

        Returns:
            np.ndarray: Environment state.

        """
        self.state = np.ones(self._obs_dim, dtype=np.uint8)
        self._lives = 5
        self.step_called = 0
        return self.state

    def step(self, action):
        """Step the environment.

        Before gym fixed overflow issue for sample() in
        np.uint8 environment, we will handle the sampling here.
        We need high=256 since np.random.uniform sample from [low, high)
        (includes low, but excludes high).

        Args:
            action (int): Action.

        Returns:
            np.ndarray: observation.
            float: reward.
            bool: terminal signal.
            dict: extra environment info.

        Raises:
            RuntimeError: step when empty lives left.

        """
        done = False
        if self.state is not None:
            # Simulating FIRE action
            if action == 1:
                if self._prev_action == 2:
                    done = True
                obs = np.full(self._obs_dim, 2, dtype=np.uint8)
            else:
                if self.random:
                    obs = np.random.uniform(low=0,
                                            high=256,
                                            size=self._obs_dim).astype(
                                                np.uint8)
                else:
                    obs = self.state + action
            if self._lives == 0:
                raise RuntimeError('DummyEnv: Cannot step when lives = 0!')
            self._lives -= 1
            if self._lives == 0:
                done = True
        else:
            raise RuntimeError(
                'DummyEnv: reset() must be called before step()!')
        self.step_called += 1
        self._prev_action = action

        return obs, 0, done, {'ale.lives': self._lives}
