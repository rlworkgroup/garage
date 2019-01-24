import unittest
import gym
import torch

from garage.contrib.exp.core.misc import get_env_spec
from garage.contrib.torch.policies.gaussian_mlp_policy import GaussianMLPPolicy


class TestGaussianMLPPolicy(unittest.TestCase):
    def test_gaussian_mlp_policy(self):
        env = gym.make('Pendulum-v0')
        spec = get_env_spec(env)
        policy = GaussianMLPPolicy(spec)

        obs = torch.Tensor(env.reset())
        obs.unsqueeze_(0)
        action = policy.sample(obs)
        logpdf = policy.logpdf(obs, action)
        # print(obs, action, logpdf)
