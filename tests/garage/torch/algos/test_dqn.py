"""Test DQN performance on cartpole."""
import copy

import pytest
import torch
from torch.nn import functional as F  # NOQA

from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import EpsilonGreedyPolicy
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import dict_np_to_torch
from garage.torch.algos import DQN
from garage.torch.policies import DiscreteQFArgmaxPolicy
from garage.torch.q_functions import DiscreteMLPQFunction
from garage.trainer import Trainer

from tests.fixtures import snapshot_config


@pytest.fixture
def setup():
    set_seed(24)
    n_epochs = 11
    steps_per_epoch = 10
    sampler_batch_size = 512
    num_timesteps = 100 * steps_per_epoch * sampler_batch_size

    env = GymEnv('CartPole-v0')

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    qf = DiscreteMLPQFunction(env_spec=env.spec, hidden_sizes=(8, 5))

    policy = DiscreteQFArgmaxPolicy(env_spec=env.spec, qf=qf)
    exploration_policy = EpsilonGreedyPolicy(env_spec=env.spec,
                                             policy=policy,
                                             total_timesteps=num_timesteps,
                                             max_epsilon=1.0,
                                             min_epsilon=0.01,
                                             decay_ratio=0.4)
    algo = DQN(env_spec=env.spec,
               policy=policy,
               qf=qf,
               exploration_policy=exploration_policy,
               replay_buffer=replay_buffer,
               steps_per_epoch=steps_per_epoch,
               qf_lr=5e-5,
               discount=0.9,
               min_buffer_size=int(1e4),
               n_train_steps=500,
               target_update_freq=30,
               buffer_batch_size=64)

    return algo, env, replay_buffer, n_epochs, sampler_batch_size


@pytest.mark.large
def test_dqn_cartpole(setup):
    trainer = Trainer(snapshot_config)
    algo, env, _, n_epochs, batch_size = setup
    trainer.setup(algo, env, sampler_cls=LocalSampler)
    last_avg_return = trainer.train(n_epochs=n_epochs, batch_size=batch_size)
    assert last_avg_return > 10
    env.close()


def test_dqn_loss(setup):
    algo, env, buff, _, batch_size = setup

    trainer = Trainer(snapshot_config)
    trainer.setup(algo, env, sampler_cls=LocalSampler)

    paths = trainer.obtain_episodes(0, batch_size=batch_size)
    buff.add_episode_batch(paths)
    samples = buff.sample_transitions(algo._buffer_batch_size)
    samples_copy = copy.deepcopy(samples)
    transitions = dict_np_to_torch(samples)

    observations = transitions['observations']
    rewards = transitions['rewards'].reshape(-1, 1)
    actions = transitions['actions']
    next_observations = transitions['next_observations']
    terminals = transitions['terminals'].reshape(-1, 1)

    next_inputs = next_observations
    inputs = observations
    with torch.no_grad():
        target_qvals = algo._target_qf(next_inputs)
        best_qvals, _ = torch.max(target_qvals, 1)
        best_qvals = best_qvals.unsqueeze(1)

    rewards_clipped = rewards
    y_target = (rewards_clipped +
                (1.0 - terminals) * algo._discount * best_qvals)
    y_target = y_target.squeeze(1)

    # optimize qf
    qvals = algo._qf(inputs)
    selected_qs = torch.sum(qvals * actions, axis=1)
    qval_loss = F.smooth_l1_loss(selected_qs, y_target)

    algo_loss, algo_targets, algo_selected_qs = algo.optimize_qf(samples_copy)
    env.close()

    assert (qval_loss.detach() == algo_loss).all()
    assert (y_target == algo_targets).all()
    assert (selected_qs == algo_selected_qs).all()
