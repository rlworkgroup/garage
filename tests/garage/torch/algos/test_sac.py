import gym
import numpy as np
import torch
from torch.nn import functional as F  # NOQA
from torch import nn as nn

from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.experiment import LocalRunner, run_experiment
from garage.replay_buffer import SimpleReplayBuffer
from garage.torch.algos import SAC
from garage.torch.q_functions import ContinuousMLPQFunction


from unittest import mock
from unittest.mock import MagicMock


class _MockDistribution(object):

    def __init__(self, action):
        self._action = action

    def rsample(self, **kwargs):
        return self._action, self._action

    def log_prob(self, value, **kwargs):
        return torch.Tensor([10.])


class DummyActorPolicy(object):

    def __init__(self,
                action=1.):
        self._action = action

    def __call__(self, observation):
        action = torch.Tensor([self._action])
        return _MockDistribution(action)

    def action(self, unused_observation):
        action = torch.Tensor([self._action], dtype=torch.float32)
        return action

    def parameters(self):
        return torch.zeros(5)


class DummyCriticNet():

    def __init__(self):
        pass

    def parameters(self):
        return torch.zeros(5)

    def __call__(self, observation, actions):
        # Biggest state is best state.
        value = torch.max(observation, dim=-1).values
        # Biggest action is best action.
        q_value = torch.max(actions, axis=-1).values
        ret = value + q_value
        return ret


class TestSacLosses():

    def testCriticLoss(self):
        policy = DummyActorPolicy()
        sac = SAC(
                  env_spec=None,
                  policy=policy,
                  qf1=DummyCriticNet(),
                  qf2=DummyCriticNet(),
                  use_automatic_entropy_tuning=True,
                  replay_buffer=None,
                  gradient_steps_per_itr=1,
                  discount=0.9,
                  buffer_batch_size=2,
                  target_entropy=3.0,
                  #initial_log_entropy=4.0,
                  optimizer=MagicMock)

        observations = torch.FloatTensor([[1, 2], [3, 4]])
        actions = torch.FloatTensor([[5], [6]])
        rewards = torch.FloatTensor([10, 20])
        terminals = [[0.], [0.]]
        next_observations = torch.FloatTensor([[5, 6], [7, 8]])
        samples = {"observation" : observations, "action" : actions, "reward" : rewards,
                    "terminal" : terminals, "next_observation" : next_observations}
        td_targets = [7.3, 19.1]
        pred_td_targets = [7., 10.]

        # Expected critic loss has factor of 2, for the two TD3 critics.
        expected_loss = 2 * F.mse_loss(
            torch.Tensor(td_targets), torch.Tensor(pred_td_targets))
        loss = sac.critic_objective(samples)
        assert(np.all(np.isclose(np.sum(loss), expected_loss)))

    def testActorLoss(self):
        policy = DummyActorPolicy()
        sac = SAC(env_spec=None,
                policy=policy,
                qf1=DummyCriticNet(),
                qf2=DummyCriticNet(),
                use_automatic_entropy_tuning=True,
                replay_buffer=None,
                discount=1,
                buffer_batch_size=2,
                target_entropy=3.0,
                initial_log_entropy=0,
                optimizer=MagicMock,
                gradient_steps_per_itr=1)

        observations = torch.Tensor([[1., 2.], [3., 4.]])
        action_dists = policy(observations)
        actions = action_dists.rsample()
        log_pi = action_dists.log_prob(actions)
        expected_loss = (2 * 10 - (2 + 1) - (4 + 1)) / 2
        loss = sac.actor_objective(observations, log_pi, actions)
        assert(np.all(np.isclose(loss,expected_loss)))

    def testAlphaLoss(self):
        policy = DummyActorPolicy()
        sac = SAC(env_spec=None,
                  policy=policy,
                  qf1=DummyCriticNet(),
                  qf2=DummyCriticNet(),
                  use_automatic_entropy_tuning=True,
                  replay_buffer=None,
                  discount=1,
                  buffer_batch_size=2,
                  target_entropy=3.0,
                  initial_log_entropy=4.0,
                  optimizer=MagicMock,
                  gradient_steps_per_itr=1)
        observations = torch.Tensor([[1., 2.], [3., 4.]])
        action_dists = policy(observations)
        actions = action_dists.rsample()
        log_pi = action_dists.log_prob(actions)
        expected_loss = 4.0 * (-10 - 3)
        loss = sac.temperature_objective(log_pi).item()
        assert(np.all(np.isclose(loss,expected_loss)))