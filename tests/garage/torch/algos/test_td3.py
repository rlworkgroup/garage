"""Test TD3 on InvertedDoublePendulum-v2."""
import pickle

import pytest
from torch.nn import functional as F

from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.np.exploration_policies import AddGaussianNoise
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import prefer_gpu
from garage.torch.algos import TD3
from garage.torch.policies import DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

from tests.fixtures import snapshot_config, TfGraphTestCase


class TestTD3(TfGraphTestCase):
    """Test class for TD3."""

    @pytest.mark.mujoco
    def test_td3_inverted_double_pendulum(self):
        deterministic.set_seed(0)
        n_epochs = 10
        steps_per_epoch = 20
        sampler_batch_size = 100
        num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size
        trainer = Trainer(snapshot_config=snapshot_config)
        env = normalize(
            GymEnv('InvertedDoublePendulum-v2', max_episode_length=100))
        policy = DeterministicMLPPolicy(env_spec=env.spec,
                                        hidden_sizes=[64, 64],
                                        hidden_nonlinearity=F.relu,
                                        output_nonlinearity=None)
        exploration_policy = AddGaussianNoise(env.spec,
                                              policy,
                                              total_timesteps=num_timesteps,
                                              max_sigma=0.1,
                                              min_sigma=0.1)
        qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                     hidden_sizes=[256, 256],
                                     hidden_nonlinearity=F.relu)
        qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                     hidden_sizes=[256, 256],
                                     hidden_nonlinearity=F.relu)
        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
        sampler = LocalSampler(agents=exploration_policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               worker_class=FragmentWorker)
        td3 = TD3(env_spec=env.spec,
                  policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  replay_buffer=replay_buffer,
                  sampler=sampler,
                  exploration_policy=exploration_policy,
                  steps_per_epoch=steps_per_epoch,
                  grad_steps_per_env_step=1,
                  num_evaluation_episodes=1,
                  discount=0.99)

        prefer_gpu()
        td3.to()
        trainer.setup(td3, env)
        trainer.train(n_epochs=n_epochs, batch_size=sampler_batch_size)

    @pytest.mark.mujoco
    def test_pickling(self):
        """Test pickle and unpickle."""

        deterministic.set_seed(0)
        n_epochs = 10
        steps_per_epoch = 20
        sampler_batch_size = 100
        num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size
        env = normalize(
            GymEnv('InvertedDoublePendulum-v2', max_episode_length=100))
        policy = DeterministicMLPPolicy(env_spec=env.spec,
                                        hidden_sizes=[64, 64],
                                        hidden_nonlinearity=F.relu,
                                        output_nonlinearity=None)
        exploration_policy = AddGaussianNoise(env.spec,
                                              policy,
                                              total_timesteps=num_timesteps,
                                              max_sigma=0.1,
                                              min_sigma=0.1)
        qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                     hidden_sizes=[256, 256],
                                     hidden_nonlinearity=F.relu)
        qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                     hidden_sizes=[256, 256],
                                     hidden_nonlinearity=F.relu)
        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
        sampler = LocalSampler(agents=exploration_policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               worker_class=FragmentWorker)
        td3 = TD3(env_spec=env.spec,
                  policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  replay_buffer=replay_buffer,
                  sampler=sampler,
                  exploration_policy=exploration_policy,
                  steps_per_epoch=steps_per_epoch,
                  grad_steps_per_env_step=1,
                  num_evaluation_episodes=1,
                  discount=0.99)
        prefer_gpu()
        td3.to()

        pickled = pickle.dumps(td3)
        unpickled = pickle.loads(pickled)
        assert unpickled
