"""Test TD3 on InvertedDoublePendulum-v2."""
import pytest
import torch

from garage.envs import GymEnv
from garage.experiment import deterministic, LocalRunner
from garage.sampler import LocalSampler
from garage.torch.algos import TD3
from garage.torch.policies import DeterministicMLPPolicy
from garage.torch.value_functions import ContinuousMLPQFunction

from tests.fixtures import snapshot_config


class TestTD3:
    """Test class for TD3."""

    @pytest.mark.mujoco
    def test_td3_inverted_double_pendulum(self):
        deterministic.set_seed(0)

        env = GymEnv('InvertedDoublePendulum-v2')
        runner = LocalRunner(snapshot_config)

        policy = DeterministicMLPPolicy(env_spec=env.spec,
                                   hidden_sizes=[64, 64],
                                   hidden_nonlinearity=torch.tanh,
                                   output_nonlinearity=None)

        qf1 = ContinuousMLPQFunction(env_spec=env.spec)
        qf2 = ContinuousMLPQFunction(env_spec=env.spec)

        td3 = TD3(env_spec=env.spec,
                   policy=policy,
                   qf1=qf1,
                   qf2=qf2
                   max_episode_length=100,
                   discount=0.99)

        runner.setup(td3, env, sampler_cls=LocalSampler)
        td3.to()
        last_avg_ret = runner.train(n_epochs=10, batch_size=100)
        assert last_avg_ret > 0

        env.close()
