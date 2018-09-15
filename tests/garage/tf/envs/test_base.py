import unittest

import gym
from nose2.tools import params

from garage.tf.envs.base import TfEnv
from tests.helpers import pickle_env_with_gym_quirks
from tests.helpers import step_env_with_gym_quirks


class TestTfEnv(unittest.TestCase):
    @params(*list(gym.envs.registry.all()))
    def test_all_gym_envs(self, spec):
        env = TfEnv(spec.make())
        step_env_with_gym_quirks(self, env, spec)

    @params(*list(gym.envs.registry.all()))
    def test_all_gym_envs_pickleable_algo(self, spec):
        env = TfEnv(env_name=spec.id)
        pickle_env_with_gym_quirks(self, env, spec, render=True)
