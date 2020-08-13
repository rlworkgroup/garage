"""This script creates a test that fails when VPG performance is too low."""
import gym
import pytest
import torch

from garage.envs import GarageEnv
from garage.experiment import deterministic, LocalRunner
from garage.sampler import LocalSampler
from garage.torch.algos import A2C
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction

from tests.fixtures import snapshot_config


class TestA2C:
    """Test class for A2C."""

    @pytest.mark.mujoco
    def test_a2c_pendulum(self):
        deterministic.set_seed(0)

        env = GarageEnv(gym.make('InvertedDoublePendulum-v2'))
        runner = LocalRunner(snapshot_config)

        policy = GaussianMLPPolicy(env_spec=env.spec,
                                   hidden_sizes=[64, 64],
                                   hidden_nonlinearity=torch.tanh,
                                   output_nonlinearity=None)

        value_function = GaussianMLPValueFunction(env_spec=env.spec)

        algo = A2C(env_spec=env.spec,
                   policy=policy,
                   value_function=value_function,
                   max_episode_length=100,
                   discount=0.99)

        runner.setup(algo, env, sampler_cls=LocalSampler)
        last_avg_ret = runner.train(n_epochs=10, batch_size=100)
        assert last_avg_ret > 0

        env.close()
