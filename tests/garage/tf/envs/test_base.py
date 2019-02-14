import unittest

import gym
from nose2.tools import params

from garage.tf.envs.base import TfEnv
from tests.helpers import step_env_with_gym_quirks


class TestTfEnv(unittest.TestCase):
    @params(*list(gym.envs.registry.all()))
    def test_all_gym_envs(self, spec):
        env = TfEnv(spec.make())
        step_env_with_gym_quirks(self, env, spec)
        env.close()

    test_all_gym_envs.cron_job = True

    @params(*list(gym.envs.registry.all()))
    def test_all_gym_envs_pickleable(self, spec):
        env = TfEnv(env_name=spec.id)
        step_env_with_gym_quirks(
            self, env, spec, n=1, render=True, serialize_env=True)
        env.close()

    test_all_gym_envs_pickleable.cron_job = True
