"""This script creates a test which fails when
 saving/resuming a model is unsuccessful."""

import tempfile

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from garage.envs import GymEnv, normalize
from garage.experiment import deterministic, SnapshotConfig
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer


@pytest.mark.mujoco
def test_torch_cpu_resume_cpu():
    """Test saving on CPU and resuming on CPU."""
    temp_dir = tempfile.TemporaryDirectory()
    snapshot_config = SnapshotConfig(snapshot_dir=temp_dir.name,
                                     snapshot_mode='last',
                                     snapshot_gap=1)
    env = normalize(GymEnv('InvertedDoublePendulum-v2',
                           max_episode_length=100))
    deterministic.set_seed(0)
    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[32, 32],
        hidden_nonlinearity=torch.nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[32, 32],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[32, 32],
                                 hidden_nonlinearity=F.relu)
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )
    trainer = Trainer(snapshot_config=snapshot_config)
    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker)
    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              sampler=sampler,
              gradient_steps_per_itr=100,
              replay_buffer=replay_buffer,
              min_buffer_size=1e3,
              target_update_tau=5e-3,
              discount=0.99,
              buffer_batch_size=64,
              reward_scale=1.,
              steps_per_epoch=2)
    sac.has_lambda = lambda x: x + 1
    trainer.setup(sac, env)
    set_gpu_mode(False)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=10, batch_size=100)
    trainer = Trainer(snapshot_config)
    trainer.restore(temp_dir.name)
    trainer.resume(n_epochs=20)
    temp_dir.cleanup()


@pytest.mark.gpu
@pytest.mark.mujoco
def test_torch_cpu_resume_gpu():
    """Test saving on CPU and resuming on GPU."""
    temp_dir = tempfile.TemporaryDirectory()
    snapshot_config = SnapshotConfig(snapshot_dir=temp_dir.name,
                                     snapshot_mode='last',
                                     snapshot_gap=1)
    env = normalize(GymEnv('InvertedDoublePendulum-v2',
                           max_episode_length=100))
    deterministic.set_seed(0)
    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[32, 32],
        hidden_nonlinearity=torch.nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[32, 32],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[32, 32],
                                 hidden_nonlinearity=F.relu)
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )
    trainer = Trainer(snapshot_config=snapshot_config)
    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker)
    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              sampler=sampler,
              gradient_steps_per_itr=100,
              replay_buffer=replay_buffer,
              min_buffer_size=1e3,
              target_update_tau=5e-3,
              discount=0.99,
              buffer_batch_size=64,
              reward_scale=1.,
              steps_per_epoch=2)
    sac.has_lambda = lambda x: x + 1
    trainer.setup(sac, env)
    set_gpu_mode(False)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=10, batch_size=100)
    trainer = Trainer(snapshot_config)
    set_gpu_mode(True)
    trainer.restore(temp_dir.name)
    trainer.resume(n_epochs=20)
    temp_dir.cleanup()


@pytest.mark.gpu
@pytest.mark.mujoco
def test_torch_gpu_resume_cpu():
    """Test saving on GPU and resuming on CPU."""
    temp_dir = tempfile.TemporaryDirectory()
    snapshot_config = SnapshotConfig(snapshot_dir=temp_dir.name,
                                     snapshot_mode='last',
                                     snapshot_gap=1)
    env = normalize(GymEnv('InvertedDoublePendulum-v2',
                           max_episode_length=100))
    deterministic.set_seed(0)
    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[32, 32],
        hidden_nonlinearity=torch.nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[32, 32],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[32, 32],
                                 hidden_nonlinearity=F.relu)
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )
    trainer = Trainer(snapshot_config=snapshot_config)
    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker)
    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              sampler=sampler,
              gradient_steps_per_itr=100,
              replay_buffer=replay_buffer,
              min_buffer_size=1e3,
              target_update_tau=5e-3,
              discount=0.99,
              buffer_batch_size=64,
              reward_scale=1.,
              steps_per_epoch=2)
    sac.has_lambda = lambda x: x + 1
    trainer.setup(sac, env)
    set_gpu_mode(True)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=10, batch_size=100)
    set_gpu_mode(False)
    trainer = Trainer(snapshot_config)
    trainer.restore(temp_dir.name)
    trainer.resume(n_epochs=20)
    temp_dir.cleanup()


@pytest.mark.gpu
@pytest.mark.mujoco
def test_torch_gpu_resume_gpu():
    """Test saving on GPU and resuming on GPU."""
    temp_dir = tempfile.TemporaryDirectory()
    snapshot_config = SnapshotConfig(snapshot_dir=temp_dir.name,
                                     snapshot_mode='last',
                                     snapshot_gap=1)
    env = normalize(GymEnv('InvertedDoublePendulum-v2',
                           max_episode_length=100))
    deterministic.set_seed(0)
    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[32, 32],
        hidden_nonlinearity=torch.nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[32, 32],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[32, 32],
                                 hidden_nonlinearity=F.relu)
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )
    trainer = Trainer(snapshot_config=snapshot_config)
    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker)
    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              sampler=sampler,
              gradient_steps_per_itr=100,
              replay_buffer=replay_buffer,
              min_buffer_size=1e3,
              target_update_tau=5e-3,
              discount=0.99,
              buffer_batch_size=64,
              reward_scale=1.,
              steps_per_epoch=2)
    sac.has_lambda = lambda x: x + 1
    trainer.setup(sac, env)
    set_gpu_mode(True)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=10, batch_size=100)
    trainer = Trainer(snapshot_config)
    trainer.restore(temp_dir.name)
    trainer.resume(n_epochs=20)
    temp_dir.cleanup()
