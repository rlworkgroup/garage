"""This script creates a test that fails when DDPG performance is too low."""
import pytest
import torch
from torch.nn import functional as F  # NOQA

from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch.algos import DDPG
from garage.torch.policies import DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

from tests.fixtures import snapshot_config


class TestDDPG:
    """Test class for DDPG."""

    @pytest.mark.mujoco_long
    def test_ddpg_double_pendulum(self):
        """Test DDPG with Pendulum environment."""
        deterministic.set_seed(0)
        trainer = Trainer(snapshot_config)
        env = GymEnv('InvertedDoublePendulum-v2', max_episode_length=100)
        policy = DeterministicMLPPolicy(env_spec=env.spec,
                                        hidden_sizes=[64, 64],
                                        hidden_nonlinearity=F.relu,
                                        output_nonlinearity=torch.tanh)

        exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec,
                                                       policy,
                                                       sigma=0.2)

        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=F.relu)

        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

        sampler = LocalSampler(agents=exploration_policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               worker_class=FragmentWorker)

        algo = DDPG(env_spec=env.spec,
                    policy=policy,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    sampler=sampler,
                    steps_per_epoch=20,
                    n_train_steps=50,
                    min_buffer_size=int(1e4),
                    exploration_policy=exploration_policy,
                    target_update_tau=1e-2,
                    discount=0.9)

        trainer.setup(algo, env)
        last_avg_ret = trainer.train(n_epochs=10, batch_size=100)
        assert last_avg_ret > 45

        env.close()

    @pytest.mark.mujoco_long
    def test_ddpg_pendulum(self):
        """Test DDPG with Pendulum environment.

        This environment has a [-3, 3] action_space bound.
        """
        deterministic.set_seed(0)
        trainer = Trainer(snapshot_config)
        env = normalize(GymEnv('InvertedPendulum-v2'))

        policy = DeterministicMLPPolicy(env_spec=env.spec,
                                        hidden_sizes=[64, 64],
                                        hidden_nonlinearity=F.relu,
                                        output_nonlinearity=torch.tanh)

        exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec,
                                                       policy,
                                                       sigma=0.2)

        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=F.relu)

        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

        sampler = LocalSampler(agents=exploration_policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               worker_class=FragmentWorker)

        algo = DDPG(env_spec=env.spec,
                    policy=policy,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    sampler=sampler,
                    steps_per_epoch=20,
                    n_train_steps=50,
                    min_buffer_size=int(1e4),
                    exploration_policy=exploration_policy,
                    target_update_tau=1e-2,
                    discount=0.9)

        trainer.setup(algo, env)
        last_avg_ret = trainer.train(n_epochs=10, batch_size=100)
        assert last_avg_ret > 10

        env.close()
