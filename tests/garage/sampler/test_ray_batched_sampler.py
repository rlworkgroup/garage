"""Tests for ray_batched_sampler."""
from unittest.mock import Mock

import numpy as np
import pytest
import ray

from garage.envs import PointEnv
from garage.envs.grid_world_env import GridWorldEnv
from garage.experiment.task_sampler import SetTaskSampler
from garage.np.policies import FixedPolicy, ScriptedPolicy
from garage.sampler import LocalSampler, RaySampler, WorkerFactory

from tests.fixtures.sampler import ray_local_session_fixture


def test_obtain_samples(ray_local_session_fixture):
    del ray_local_session_fixture
    env = GridWorldEnv(desc='4x4')
    policy = ScriptedPolicy(
        scripted_actions=[2, 2, 1, 0, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1])
    algo = Mock(env_spec=env.spec, policy=policy, max_episode_length=16)

    assert ray.is_initialized()
    workers = WorkerFactory(seed=100,
                            max_episode_length=algo.max_episode_length,
                            n_workers=8)
    sampler1 = RaySampler.from_worker_factory(workers, policy, env)
    sampler2 = LocalSampler.from_worker_factory(workers, policy, env)
    eps1 = sampler1.obtain_samples(0, 1000,
                                   tuple(algo.policy.get_param_values()))
    eps2 = sampler2.obtain_samples(0, 1000,
                                   tuple(algo.policy.get_param_values()))

    assert eps1.observations.shape[0] >= 1000
    assert eps1.actions.shape[0] >= 1000
    assert (sum(eps1.rewards[:eps1.lengths[0]]) == sum(
        eps2.rewards[:eps2.lengths[0]]) == 1)

    true_obs = np.array([0, 1, 2, 6, 10, 14])
    true_actions = np.array([2, 2, 1, 1, 1, 2])
    true_rewards = np.array([0, 0, 0, 0, 0, 1])
    start = 0
    for length in eps1.lengths:
        observations = eps1.observations[start:start + length]
        actions = eps1.actions[start:start + length]
        rewards = eps1.rewards[start:start + length]
        assert np.array_equal(observations, true_obs)
        assert np.array_equal(actions, true_actions)
        assert np.array_equal(rewards, true_rewards)
        start += length
    sampler1.shutdown_worker()
    sampler2.shutdown_worker()
    env.close()


def test_update_envs_env_update(ray_local_session_fixture):
    del ray_local_session_fixture
    assert ray.is_initialized()
    max_episode_length = 16
    env = PointEnv()
    policy = FixedPolicy(env.spec,
                         scripted_actions=[
                             env.action_space.sample()
                             for _ in range(max_episode_length)
                         ])
    tasks = SetTaskSampler(PointEnv)
    n_workers = 8
    workers = WorkerFactory(seed=100,
                            max_episode_length=max_episode_length,
                            n_workers=n_workers)
    sampler = RaySampler.from_worker_factory(workers, policy, env)
    episodes = sampler.obtain_samples(0,
                                      160,
                                      np.asarray(policy.get_param_values()),
                                      env_update=tasks.sample(n_workers))
    mean_rewards = []
    goals = []
    for eps in episodes.split():
        mean_rewards.append(eps.rewards.mean())
        goals.append(eps.env_infos['task'][0]['goal'])
    assert np.var(mean_rewards) > 0
    assert np.var(goals) > 0
    with pytest.raises(ValueError):
        sampler.obtain_samples(0,
                               10,
                               np.asarray(policy.get_param_values()),
                               env_update=tasks.sample(n_workers + 1))


def test_obtain_exact_episodes(ray_local_session_fixture):
    del ray_local_session_fixture
    assert ray.is_initialized()
    max_episode_length = 15
    n_workers = 8
    env = PointEnv()
    per_worker_actions = [env.action_space.sample() for _ in range(n_workers)]
    policies = [
        FixedPolicy(env.spec, [action] * max_episode_length)
        for action in per_worker_actions
    ]
    workers = WorkerFactory(seed=100,
                            max_episode_length=max_episode_length,
                            n_workers=n_workers)
    sampler = RaySampler.from_worker_factory(workers, policies, envs=env)
    n_eps_per_worker = 3
    episodes = sampler.obtain_exact_episodes(n_eps_per_worker, policies)
    # At least one action per episode.
    assert sum(episodes.lengths) >= n_workers * n_eps_per_worker
    # All of the episodes.
    assert len(episodes.lengths) == n_workers * n_eps_per_worker
    worker = -1
    for count, eps in enumerate(episodes.split()):
        if count % n_eps_per_worker == 0:
            worker += 1
        assert (eps.actions == per_worker_actions[worker]).all()


def test_init_with_env_updates(ray_local_session_fixture):
    del ray_local_session_fixture
    assert ray.is_initialized()
    max_episode_length = 16
    env = PointEnv()
    policy = FixedPolicy(env.spec,
                         scripted_actions=[
                             env.action_space.sample()
                             for _ in range(max_episode_length)
                         ])
    tasks = SetTaskSampler(PointEnv)
    n_workers = 8
    workers = WorkerFactory(seed=100,
                            max_episode_length=max_episode_length,
                            n_workers=n_workers)
    sampler = RaySampler.from_worker_factory(workers,
                                             policy,
                                             envs=tasks.sample(n_workers))
    episodes = sampler.obtain_samples(0, 160, policy)
    assert sum(episodes.lengths) >= 160
