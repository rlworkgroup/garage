import unittest

import gym
import numpy as np

from garage.baselines import ZeroBaseline
from garage.envs import Step
from garage.theano.algos import TRPO
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy


class DummyEnv(gym.Env):
    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-5.0, high=5.0, shape=(1, ), dtype=np.float32)

    def reset(self):
        return np.zeros(1)

    def step(self, action):
        return Step(
            observation=np.zeros(1), reward=np.random.normal(), done=True)


class TestTRPO(unittest.TestCase):
    def test_trpo_relu_nan(self):
        env = TheanoEnv(DummyEnv())
        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(1, ))
        baseline = ZeroBaseline(env_spec=env.spec)
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            n_itr=1,
            batch_size=1000,
            max_path_length=100,
            step_size=0.001)
        algo.train()
        assert not np.isnan(np.sum(policy.get_param_values()))

    def test_trpo_deterministic_nan(self):
        env = TheanoEnv(DummyEnv())
        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(1, ))
        policy._l_log_std.param.set_value([np.float32(np.log(1e-8))])
        baseline = ZeroBaseline(env_spec=env.spec)
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            n_itr=10,
            batch_size=1000,
            max_path_length=100,
            step_size=0.01)
        algo.train()
        assert not np.isnan(np.sum(policy.get_param_values()))
