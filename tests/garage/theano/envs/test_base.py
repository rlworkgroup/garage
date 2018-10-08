import unittest

import gym
from nose2.tools import params

from garage.theano.envs.base import TheanoEnv
from tests.helpers import step_env_with_gym_quirks


class TestTheanoEnv(unittest.TestCase):
    @params(*list(gym.envs.registry.all()))
    def test_all_gym_envs(self, spec):
        env = TheanoEnv(spec.make())
        step_env_with_gym_quirks(self, env, spec, render=True)

    test_all_gym_envs.cron_job = True

    @params(*list(gym.envs.registry.all()))
    def test_all_gym_envs_pickleable(self, spec):
        env = TheanoEnv(env_name=spec.id)
        step_env_with_gym_quirks(
            self, env, spec, n=1, render=True, serialize_env=True)

    test_all_gym_envs_pickleable.cron_job = True
