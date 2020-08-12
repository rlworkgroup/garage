"""Module for testing SAC loss functions."""
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from garage.envs import GymEnv, normalize
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction

from tests.fixtures import snapshot_config


class _MockDistribution:

    def __init__(self, action):
        self._action = action

    def rsample_with_pre_tanh_value(self, **kwargs):
        del kwargs
        return self._action, self._action

    def rsample(self, **kwargs):
        del kwargs
        return self._action, self._action

    def log_prob(self, value, **kwargs):
        del kwargs
        del value
        return torch.Tensor([10.])


class DummyActorPolicy:
    """Dummy Policy Network."""

    def __init__(self, action=1.):
        self._action = action

    def __call__(self, observation):
        """Dummy forward operation. Returns a dummy distribution."""
        action = torch.Tensor([self._action])
        return _MockDistribution(action), {}

    def action(self, unused_observation):
        """Dummy action function. Always returns 1."""
        del unused_observation
        action = torch.Tensor([self._action], dtype=torch.float32)
        return action

    def parameters(self):
        """Mock Params function, returns all 0s."""
        return torch.zeros(5)


class DummyCriticNet:
    """Mock QFunction."""

    def __init__(self):
        pass

    def parameters(self):
        """Mock Params function, returns all 0s."""
        return torch.zeros(5)

    def __call__(self, observation, actions):
        """Mock Sampling function."""
        # Biggest state is best state.
        value = torch.max(observation, dim=-1).values
        # Biggest action is best action.
        q_value = torch.max(actions, axis=-1).values
        ret = value + q_value
        return ret


def testCriticLoss():
    """Test Sac Critic/QF loss."""
    # pylint: disable=no-member
    policy = DummyActorPolicy()
    sac = SAC(env_spec=None,
              policy=policy,
              qf1=DummyCriticNet(),
              qf2=DummyCriticNet(),
              replay_buffer=None,
              gradient_steps_per_itr=1,
              discount=0.9,
              buffer_batch_size=2,
              target_entropy=3.0,
              max_episode_length=10,
              optimizer=MagicMock)

    observations = torch.FloatTensor([[1, 2], [3, 4]])
    actions = torch.FloatTensor([[5], [6]])
    rewards = torch.FloatTensor([10, 20])
    terminals = torch.Tensor([[0.], [0.]])
    next_observations = torch.FloatTensor([[5, 6], [7, 8]])
    samples_data = {
        'observation': observations,
        'action': actions,
        'reward': rewards,
        'terminal': terminals,
        'next_observation': next_observations
    }
    td_targets = [7.3, 19.1]
    pred_td_targets = [7., 10.]

    # Expected critic loss has factor of 2, for the two TD3 critics.
    expected_loss = 2 * F.mse_loss(torch.Tensor(td_targets),
                                   torch.Tensor(pred_td_targets))
    loss = sac._critic_objective(samples_data)
    assert np.all(np.isclose(np.sum(loss), expected_loss))


def testActorLoss():
    """Test Sac Actor/Policy loss."""
    # pylint: disable=no-member
    policy = DummyActorPolicy()
    sac = SAC(env_spec=None,
              policy=policy,
              qf1=DummyCriticNet(),
              qf2=DummyCriticNet(),
              replay_buffer=None,
              discount=1,
              buffer_batch_size=2,
              target_entropy=3.0,
              initial_log_entropy=0,
              optimizer=MagicMock,
              max_episode_length=10,
              gradient_steps_per_itr=1)

    observations = torch.Tensor([[1., 2.], [3., 4.]])
    action_dists = policy(observations)[0]
    actions = torch.Tensor(action_dists.rsample_with_pre_tanh_value())
    samples_data = dict(observation=observations)
    log_pi = action_dists.log_prob(actions)
    expected_loss = (2 * 10 - (2 + 1) - (4 + 1)) / 2
    loss = sac._actor_objective(samples_data, actions, log_pi)
    assert np.all(np.isclose(loss, expected_loss))


def testTemperatureLoss():
    """Test Sac temperature loss."""
    # pylint: disable=no-member
    policy = DummyActorPolicy()
    sac = SAC(env_spec=None,
              policy=policy,
              qf1=DummyCriticNet(),
              qf2=DummyCriticNet(),
              replay_buffer=None,
              discount=1,
              buffer_batch_size=2,
              target_entropy=3.0,
              initial_log_entropy=4.0,
              optimizer=MagicMock,
              max_episode_length=10,
              gradient_steps_per_itr=1)
    observations = torch.Tensor([[1., 2.], [3., 4.]])
    action_dists = policy(observations)[0]
    actions = action_dists.rsample_with_pre_tanh_value()
    log_pi = action_dists.log_prob(actions)
    samples_data = dict(observation=observations, action=actions)
    expected_loss = 4.0 * (-10 - 3)
    loss = sac._temperature_objective(log_pi, samples_data).item()
    assert np.all(np.isclose(loss, expected_loss))


@pytest.mark.mujoco
def test_sac_inverted_double_pendulum():
    """Test Sac performance on inverted pendulum."""
    # pylint: disable=unexpected-keyword-arg
    env = normalize(GymEnv('InvertedDoublePendulum-v2'))
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
    runner = LocalRunner(snapshot_config=snapshot_config)
    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              gradient_steps_per_itr=100,
              max_episode_length=100,
              replay_buffer=replay_buffer,
              min_buffer_size=1e3,
              target_update_tau=5e-3,
              discount=0.99,
              buffer_batch_size=64,
              reward_scale=1.,
              steps_per_epoch=2)
    runner.setup(sac, env, sampler_cls=LocalSampler)
    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    sac.to()
    ret = runner.train(n_epochs=12, batch_size=200, plot=False)
    # check that automatic entropy tuning is used
    assert sac._use_automatic_entropy_tuning
    # assert that there was a gradient properly connected to alpha
    # this doesn't verify that the path from the temperature objective is
    # correct.
    assert not torch.allclose(torch.Tensor([1.]), sac._log_alpha.to('cpu'))
    # check that policy is learning beyond predecided threshold
    assert ret > 80


@pytest.mark.mujoco
def test_fixed_alpha():
    """Test if using fixed_alpha ensures that alpha is non differentiable."""
    # pylint: disable=unexpected-keyword-arg
    env = normalize(GymEnv('InvertedDoublePendulum-v2'))
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
    runner = LocalRunner(snapshot_config=snapshot_config)
    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              gradient_steps_per_itr=100,
              max_episode_length=100,
              replay_buffer=replay_buffer,
              min_buffer_size=100,
              target_update_tau=5e-3,
              discount=0.99,
              buffer_batch_size=64,
              reward_scale=1.,
              steps_per_epoch=1,
              fixed_alpha=np.exp(0.5))
    runner.setup(sac, env, sampler_cls=LocalSampler)
    sac.to()
    runner.train(n_epochs=1, batch_size=100, plot=False)
    assert torch.allclose(torch.Tensor([0.5]), sac._log_alpha.cpu())
    assert not sac._use_automatic_entropy_tuning
