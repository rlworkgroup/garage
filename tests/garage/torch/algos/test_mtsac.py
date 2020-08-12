"""Module for testing MTSAC."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from garage.envs import GymEnv, MultiEnvWrapper
from garage.envs.multi_env_wrapper import round_robin_strategy
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import global_device, set_gpu_mode
from garage.torch.algos import MTSAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction

from tests.fixtures import snapshot_config


@pytest.mark.mujoco
def test_mtsac_get_log_alpha(monkeypatch):
    """Check that the private function _get_log_alpha functions correctly.

    MTSAC uses disentangled alphas, meaning that

    """
    env_names = ['CartPole-v0', 'CartPole-v1']
    task_envs = [GymEnv(name) for name in env_names]
    env = MultiEnvWrapper(task_envs, sample_strategy=round_robin_strategy)
    deterministic.set_seed(0)
    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[1, 1],
        hidden_nonlinearity=torch.nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[1, 1],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[1, 1],
                                 hidden_nonlinearity=F.relu)
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )

    num_tasks = 2
    buffer_batch_size = 2
    mtsac = MTSAC(policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  gradient_steps_per_itr=150,
                  max_episode_length=150,
                  eval_env=env,
                  env_spec=env.spec,
                  num_tasks=num_tasks,
                  steps_per_epoch=5,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1e3,
                  target_update_tau=5e-3,
                  discount=0.99,
                  buffer_batch_size=buffer_batch_size)
    monkeypatch.setattr(mtsac, '_log_alpha', torch.Tensor([1., 2.]))
    for i, _ in enumerate(env_names):
        obs = torch.Tensor([env.reset()[0]] * buffer_batch_size)
        log_alpha = mtsac._get_log_alpha(dict(observation=obs))
        assert (log_alpha == torch.Tensor([i + 1, i + 1])).all().item()
        assert log_alpha.size() == torch.Size([mtsac._buffer_batch_size])


@pytest.mark.mujoco
def test_mtsac_inverted_double_pendulum():
    """Performance regression test of MTSAC on 2 InvDoublePendulum envs."""
    env_names = ['InvertedDoublePendulum-v2', 'InvertedDoublePendulum-v2']
    task_envs = [GymEnv(name) for name in env_names]
    env = MultiEnvWrapper(task_envs, sample_strategy=round_robin_strategy)
    test_envs = MultiEnvWrapper(task_envs,
                                sample_strategy=round_robin_strategy)
    deterministic.set_seed(0)
    runner = LocalRunner(snapshot_config=snapshot_config)
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
    num_tasks = 2
    buffer_batch_size = 128
    mtsac = MTSAC(policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  gradient_steps_per_itr=100,
                  max_episode_length=100,
                  eval_env=test_envs,
                  env_spec=env.spec,
                  num_tasks=num_tasks,
                  steps_per_epoch=5,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1e3,
                  target_update_tau=5e-3,
                  discount=0.99,
                  buffer_batch_size=buffer_batch_size)
    runner.setup(mtsac, env, sampler_cls=LocalSampler)
    ret = runner.train(n_epochs=8, batch_size=128, plot=False)
    assert ret > 0


def test_to():
    """Test the torch function that moves modules to GPU.

        Test that the policy and qfunctions are moved to gpu if gpu is
        available.

    """
    env_names = ['CartPole-v0', 'CartPole-v1']
    task_envs = [GymEnv(name) for name in env_names]
    env = MultiEnvWrapper(task_envs, sample_strategy=round_robin_strategy)
    deterministic.set_seed(0)
    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[1, 1],
        hidden_nonlinearity=torch.nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[1, 1],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[1, 1],
                                 hidden_nonlinearity=F.relu)
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )

    num_tasks = 2
    buffer_batch_size = 2
    mtsac = MTSAC(policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  gradient_steps_per_itr=150,
                  max_episode_length=150,
                  eval_env=env,
                  env_spec=env.spec,
                  num_tasks=num_tasks,
                  steps_per_epoch=5,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1e3,
                  target_update_tau=5e-3,
                  discount=0.99,
                  buffer_batch_size=buffer_batch_size)

    set_gpu_mode(torch.cuda.is_available())
    mtsac.to()
    device = global_device()
    for param in mtsac._qf1.parameters():
        assert param.device == device
    for param in mtsac._qf2.parameters():
        assert param.device == device
    for param in mtsac._qf2.parameters():
        assert param.device == device
    for param in mtsac.policy.parameters():
        assert param.device == device
    assert mtsac._log_alpha.device == device


@pytest.mark.mujoco
def test_fixed_alpha():
    """Test if using fixed_alpha ensures that alpha is non differentiable."""
    env_names = ['InvertedDoublePendulum-v2', 'InvertedDoublePendulum-v2']
    task_envs = [GymEnv(name) for name in env_names]
    env = MultiEnvWrapper(task_envs, sample_strategy=round_robin_strategy)
    test_envs = MultiEnvWrapper(task_envs,
                                sample_strategy=round_robin_strategy)
    deterministic.set_seed(0)
    runner = LocalRunner(snapshot_config=snapshot_config)
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
    num_tasks = 2
    buffer_batch_size = 128
    mtsac = MTSAC(policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  gradient_steps_per_itr=100,
                  max_episode_length=100,
                  eval_env=test_envs,
                  env_spec=env.spec,
                  num_tasks=num_tasks,
                  steps_per_epoch=1,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1e3,
                  target_update_tau=5e-3,
                  discount=0.99,
                  buffer_batch_size=buffer_batch_size,
                  fixed_alpha=np.exp(0.5))
    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    mtsac.to()
    assert torch.allclose(torch.Tensor([0.5] * num_tasks),
                          mtsac._log_alpha.to('cpu'))
    runner.setup(mtsac, env, sampler_cls=LocalSampler)
    runner.train(n_epochs=1, batch_size=128, plot=False)
    assert torch.allclose(torch.Tensor([0.5] * num_tasks),
                          mtsac._log_alpha.to('cpu'))
    assert not mtsac._use_automatic_entropy_tuning
