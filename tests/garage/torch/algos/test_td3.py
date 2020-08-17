"""Test TD3 on InvertedDoublePendulum-v2."""
import pytest
import gym
import torch
from torch.nn import functional as F

from garage.envs import GarageEnv, normalize
from garage.experiment import deterministic, LocalRunner
from garage.sampler import LocalSampler
from garage.experiment.deterministic import set_seed
from garage.replay_buffer import PathBuffer
from garage.torch.algos import TD3
from garage.torch.policies import DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction

from tests.fixtures import snapshot_config


class TestTD3:
    """Test class for TD3."""

    @pytest.mark.mujoco
    def test_td3_inverted_double_pendulum(self):
        deterministic.set_seed(0)

        env = GarageEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
        runner = LocalRunner(snapshot_config)

        policy = DeterministicMLPPolicy(env_spec=env.spec,
                                   hidden_sizes=[64, 64],
                                   hidden_nonlinearity=F.relu,
                                   output_nonlinearity=None)

        qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

        qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

        td3 = TD3(env_spec=env.spec,
                   policy=policy,
                   qf1=qf1,
                   qf2=qf2,
                   replay_buffer=replay_buffer,
                   steps_per_epoch=20,
                   max_episode_length=1000,
                   grad_steps_per_env_step=1,
                   discount=0.99)

        runner.setup(td3, env, sampler_cls=LocalSampler)
        last_avg_ret = runner.train(n_epochs=10, batch_size=100)
        print(last_avg_ret, type(last_avg_ret))
        assert last_avg_ret > 0

        env.close()
