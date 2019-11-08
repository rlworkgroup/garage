import gym
import torch
from torch.nn import functional as F  # NOQA
from torch import nn as nn

from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.experiment import LocalRunner, run_experiment
from garage.replay_buffer import SimpleReplayBuffer
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


class _MockDistribution(object):

  def __init__(self, action):
    self._action = action

  def sample(self):
    return self._action

  def log_prob(self, unused_sample):
    return torch.Tensor(10., shape=[1])


class DummyActorPolicy(object):

  def __init__(self,
               time_step_spec,
               action_spec,
               actor_network,
               training=False):
    del time_step_spec
    del actor_network
    del training
    single_action_spec = tf.nest.flatten(action_spec)[0]
    # Action is maximum of action range.
    self._action = single_action_spec.maximum
    self._action_spec = action_spec

  def action(self, time_step):
    del time_step
    action = torch.Tensor(self._action, dtype=torch.float32, shape=[1])
    return PolicyStep(action=action)

  def distribution(self, time_step, policy_state=()):
    del policy_state
    action = self.action(time_step).action
    return PolicyStep(action=_MockDistribution(action))

  def get_initial_state(self, batch_size):
    del batch_size
    return ()


class DummyCriticNet(network.Network):

  def __init__(self):
    super(DummyCriticNet, self).__init__(
        input_tensor_spec=(tensor_spec.TensorSpec([], torch.float32),
                           tensor_spec.TensorSpec([], torch.float32)),
        state_spec=(), name=None)

  def copy(self, name=''):
    del name
    return copy.copy(self)

  def call(self, inputs, step_type, network_state=()):
    del step_type
    del network_state
    observation, actions = inputs
    actions = tf.cast(tf.nest.flatten(actions)[0], torch.float32)

    states = tf.cast(tf.nest.flatten(observation)[0], torch.float32)
    # Biggest state is best state.
    value = tf.reduce_max(input_tensor=states, axis=-1)
    value = tf.reshape(value, [-1])

    # Biggest action is best action.
    q_value = tf.reduce_max(input_tensor=actions, axis=-1)
    q_value = tf.reshape(q_value, [-1])
    # Biggest state is best state.
    return value + q_value, ()



class TestSacLosses():

    def setUp(self):
        super(TestSacLosses, self).setUp()
        self._obs_spec = torch.Tensor([2], dtype=torch.float32)
        self._time_step_spec = ts.time_step_spec(self._obs_spec)
        self._action_spec = tensor_spec.BoundedTensorSpec([1], torch.float32, -1, 1)

    def testCreateAgent(self):
        pass

    def testCriticLoss(self):
        agent = sac_agent.SacAgent(
            self._time_step_spec,
            self._action_spec,
            critic_network=DummyCriticNet(),
            actor_network=None,
            actor_optimizer=None,
            critic_optimizer=None,
            alpha_optimizer=None,
            actor_policy_ctor=DummyActorPolicy)

        observations = torch.Tensor([[1, 2], [3, 4]], dtype=torch.float32)
        # time_steps = ts.restart(observations)
        actions = torch.Tensor([[5], [6]], dtype=torch.float32)
        rewards = torch.Tensor([10, 20], dtype=torch.float32)
        discounts = torch.Tensor([0.9, 0.9], dtype=torch.float32)
        next_observations = torch.Tensor([[5, 6], [7, 8]], dtype=torch.float32)
        # next_time_steps = ts.transition(next_observations, rewards, discounts)


        td_targets = [7.3, 19.1]
        pred_td_targets = [7., 10.]

        self.evaluate(tf.compat.v1.global_variables_initializer())

        # Expected critic loss has factor of 2, for the two TD3 critics.
        expected_loss = self.evaluate(2 * tf.compat.v1.losses.mean_squared_error(
            torch.Tensor(td_targets), torch.Tensor(pred_td_targets)))

        loss = agent.critic_loss(
            time_steps,
            actions,
            next_time_steps,
            td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        loss_ = self.evaluate(loss)
        self.assertAllClose(loss_, expected_loss)

    def testActorLoss(self):
        agent = sac_agent.SacAgent(
            self._time_step_spec,
            self._action_spec,
            critic_network=DummyCriticNet(),
            actor_network=None,
            actor_optimizer=None,
            critic_optimizer=None,
            alpha_optimizer=None,
            actor_policy_actor=DummyActorPolicy)

        observations = torch.Tensor([[1, 2], [3, 4]], dtype=torch.float32)
        time_steps = ts.restart(observations, batch_size=2) #returns the rewards = 0 and discount = 1 for these obs along with step_type.first
        # the action returned by the mock policy will always be [1.]*num of observations inputed

        expected_loss = (2 * 10 - (2 + 1) - (4 + 1)) / 2
        loss = agent.actor_loss(time_steps)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        loss_ = self.evaluate(loss)
        self.assertAllClose(loss_, expected_loss)

    def testAlphaLoss(self):
        # initial_log_alpha = 4.0
        # target_entropy = 3.0
        sac = SAC(env_spec=None,
                  policy=DummyActorPolicy,
                  qf1=DummyCriticNet,
                  qf2=DummyCriticNet,
                  target_qf1=DummyCriticNet,
                  target_qf2=DummyCriticNet,
                  use_automatic_entropy_tuning=True,
                  replay_buffer=None,
                  min_buffer_size=1e3,
                  target_update_tau=5e-3,
                  discount=1,
                  buffer_batch_size=2,
                  target_entropy=3.0,
                  initial_log_entropy=4.0)
        observations = torch.Tensor([[1, 2], [3, 4]], dtype=torch.float32)

        time_steps = ts.restart(observations, batch_size=2) #returns the rewards and discount for these obs along with step_type.first

        # tf agents mock policy will always return a log_pi = 10.0

        expected_loss = 4.0 * (-10 - 3)

        # alpha loss takes the rewards and discount, calls 
        loss = agent.alpha_loss(time_steps) # get actions and log_pis based on the collected actions
                                            # based on policy architecture, e.g. what the hell does the policy return on this observation
        loss_ = self.evaluate(loss)
        self.assertAllClose(loss_, expected_loss)

if __name__ == '__main__':
    tf.test.main()