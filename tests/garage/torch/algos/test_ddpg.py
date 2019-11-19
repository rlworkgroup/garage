"""
This script creates a test that fails when garage.tf.algos.DDPG performance is
too low.
"""
import gym
import pytest
import torch
from torch.nn import functional as F  # NOQA

from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.experiment import LocalRunner
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.torch.algos import DDPG
from garage.torch.policies import DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from tests.fixtures import snapshot_config


class TestDDPG:

    @pytest.mark.large
    def test_ddpg_double_pendulum(self):
        """Test DDPG with Pendulum environment."""
        runner = LocalRunner(snapshot_config)
        env = GarageEnv(gym.make('InvertedDoublePendulum-v2'))
        action_noise = OUStrategy(env.spec, sigma=0.2)

        policy = DeterministicMLPPolicy(env_spec=env.spec,
                                        hidden_sizes=[64, 64],
                                        hidden_nonlinearity=F.relu,
                                        output_nonlinearity=torch.tanh)

        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=F.relu)

        replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                           size_in_transitions=int(1e6),
                                           time_horizon=100)

        algo = DDPG(env_spec=env.spec,
                    policy=policy,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    n_train_steps=50,
                    min_buffer_size=int(1e4),
                    exploration_strategy=action_noise,
                    target_update_tau=1e-2,
                    policy_lr=1e-4,
                    qf_lr=1e-3,
                    discount=0.9)

        runner.setup(algo, env)
        last_avg_ret = runner.train(n_epochs=10,
                                    n_epoch_cycles=20,
                                    batch_size=100)
        assert last_avg_ret > 45

        env.close()

    @pytest.mark.large
    def test_ddpg_pendulum(self):
        """
        Test DDPG with Pendulum environment.

        This environment has a [-3, 3] action_space bound.
        """
        runner = LocalRunner(snapshot_config)
        env = GarageEnv(normalize(gym.make('InvertedPendulum-v2')))
        action_noise = OUStrategy(env.spec, sigma=0.2)

        policy = DeterministicMLPPolicy(env_spec=env.spec,
                                        hidden_sizes=[64, 64],
                                        hidden_nonlinearity=F.relu,
                                        output_nonlinearity=torch.tanh)

        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=F.relu)

        replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                           size_in_transitions=int(1e6),
                                           time_horizon=100)

        algo = DDPG(env_spec=env.spec,
                    policy=policy,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    n_train_steps=50,
                    min_buffer_size=int(1e4),
                    exploration_strategy=action_noise,
                    target_update_tau=1e-2,
                    policy_lr=1e-4,
                    qf_lr=1e-3,
                    discount=0.9)

        runner.setup(algo, env)
        last_avg_ret = runner.train(n_epochs=10,
                                    n_epoch_cycles=20,
                                    batch_size=100)
        assert last_avg_ret > 10

        env.close()
