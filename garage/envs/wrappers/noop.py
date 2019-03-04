"""Noop wrapper for gym.Env."""
import gym
import numpy as np


class Noop(gym.Wrapper):
    """
    Noop wrapper for gym.Env.

    It samples initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    Args:
        env: The environment to be wrapped.
        noop_max: Maximum number no-op to be performed on reset.
    """

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert noop_max > 0, "Noop_max should be larger than 0!"
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP', (
            "No-op should be the 0-th action but it's not in {}!".format(env))

    def reset(self, **kwargs):
        """gym.Env reset function."""
        obs = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs
