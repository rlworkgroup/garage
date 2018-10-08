import gym
import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.envs.util import flat_dim
from garage.misc import autoargs
from garage.misc.overrides import overrides


class NoisyObservationEnv(gym.Wrapper, Serializable):
    @autoargs.arg(
        'obs_noise',
        type=float,
        help='Noise added to the observations (note: this makes the '
        'problem non-Markovian!)')
    def __init__(
            self,
            env,
            obs_noise=1e-1,
    ):
        super().__init__(env)

        self.obs_noise = obs_noise
        self._action_flat_dim = flat_dim(self.action_space)

        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())

    def get_obs_noise_scale_factor(self, obs):
        # return np.abs(obs)
        return np.ones_like(obs)

    def inject_obs_noise(self, obs):
        """
        Inject entry-wise noise to the observation. This should not change
        the dimension of the observation.
        """
        noise = self.get_obs_noise_scale_factor(obs) * self.obs_noise * \
            np.random.normal(size=obs.shape)
        return obs + noise

    def get_current_obs(self):
        return self.inject_obs_noise(self.env.get_current_obs())

    @overrides
    def reset(self):
        obs = self.env.reset()
        return self.inject_obs_noise(obs)

    @overrides
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(self.inject_obs_noise(next_obs), reward, done, **info)


class DelayedActionEnv(gym.Wrapper, Serializable):
    @autoargs.arg(
        'action_delay', type=int, help='Time steps before action is realized')
    def __init__(
            self,
            env,
            action_delay=3,
    ):
        assert action_delay > 0, "Should not use this env transformer"
        Serializable.quick_init(self, locals())
        super(DelayedActionEnv, self).__init__(env)
        self.action_delay = action_delay
        self._action_flat_dim = flat_dim(self.action_space)
        self._queued_actions = None

    @overrides
    def reset(self):
        obs = self.env.reset()
        self._queued_actions = np.zeros(
            self.action_delay * self._action_flat_dim)
        return obs

    @overrides
    def step(self, action):
        queued_action = self._queued_actions[:self._action_flat_dim]
        next_obs, reward, done, info = self.env.step(queued_action)
        self._queued_actions = np.concatenate(
            [self._queued_actions[self._action_flat_dim:], action])
        return Step(next_obs, reward, done, **info)
