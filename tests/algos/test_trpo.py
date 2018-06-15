import gym
import lasagne.nonlinearities
import numpy as np
import theano.tensor as TT

from garage.algos import TRPO
from garage.baselines import ZeroBaseline
from garage.envs import Step
from garage.policies import GaussianMLPPolicy


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


def naive_relu(x):
    return TT.max(x, 0)


def test_trpo_relu_nan():
    env = DummyEnv()
    policy = GaussianMLPPolicy(
        env_spec=env.spec, hidden_nonlinearity=naive_relu, hidden_sizes=(1, ))
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


def test_trpo_deterministic_nan():
    env = DummyEnv()
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
