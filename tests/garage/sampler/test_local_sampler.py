import numpy as np
import pytest

from garage.envs import GarageEnv, PointEnv
from garage.experiment.task_sampler import SetTaskSampler
from garage.np.policies import FixedPolicy, ScriptedPolicy
from garage.sampler import LocalSampler, WorkerFactory


def test_update_envs_env_update():
    max_episode_length = 16
    env = GarageEnv(PointEnv())
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
    sampler = LocalSampler.from_worker_factory(workers, policy, env)
    rollouts = sampler.obtain_samples(0,
                                      161,
                                      np.asarray(policy.get_param_values()),
                                      env_update=tasks.sample(n_workers))
    mean_rewards = []
    goals = []
    for rollout in rollouts.split():
        mean_rewards.append(rollout.rewards.mean())
        goals.append(rollout.env_infos['task'][0]['goal'])
    assert len(mean_rewards) == 11
    assert len(goals) == 11
    assert np.var(mean_rewards) > 1e-2
    assert np.var(goals) > 1e-2
    with pytest.raises(ValueError):
        sampler.obtain_samples(0,
                               10,
                               np.asarray(policy.get_param_values()),
                               env_update=tasks.sample(n_workers + 1))


def test_init_with_env_updates():
    max_episode_length = 16
    env = GarageEnv(PointEnv())
    policy = FixedPolicy(env.spec,
                         scripted_actions=[
                             env.action_space.sample()
                             for _ in range(max_episode_length)
                         ])
    tasks = SetTaskSampler(lambda: GarageEnv(PointEnv()))
    n_workers = 8
    workers = WorkerFactory(seed=100,
                            max_episode_length=max_episode_length,
                            n_workers=n_workers)
    sampler = LocalSampler.from_worker_factory(workers,
                                               policy,
                                               envs=tasks.sample(n_workers))
    rollouts = sampler.obtain_samples(0, 160, policy)
    assert sum(rollouts.lengths) >= 160


def test_obtain_exact_trajectories():
    max_episode_length = 15
    n_workers = 8
    env = GarageEnv(PointEnv())
    per_worker_actions = [env.action_space.sample() for _ in range(n_workers)]
    policies = [
        FixedPolicy(env.spec, [action] * max_episode_length)
        for action in per_worker_actions
    ]
    workers = WorkerFactory(seed=100,
                            max_episode_length=max_episode_length,
                            n_workers=n_workers)
    sampler = LocalSampler.from_worker_factory(workers, policies, envs=env)
    n_traj_per_worker = 3
    rollouts = sampler.obtain_exact_trajectories(n_traj_per_worker,
                                                 agent_update=policies)
    # At least one action per trajectory.
    assert sum(rollouts.lengths) >= n_workers * n_traj_per_worker
    # All of the trajectories.
    assert len(rollouts.lengths) == n_workers * n_traj_per_worker
    worker = -1
    for count, rollout in enumerate(rollouts.split()):
        if count % n_traj_per_worker == 0:
            worker += 1
        assert (rollout.actions == per_worker_actions[worker]).all()


def test_no_seed():
    max_episode_length = 16
    env = GarageEnv(PointEnv())
    policy = FixedPolicy(env.spec,
                         scripted_actions=[
                             env.action_space.sample()
                             for _ in range(max_episode_length)
                         ])
    n_workers = 8
    workers = WorkerFactory(seed=None,
                            max_episode_length=max_episode_length,
                            n_workers=n_workers)
    sampler = LocalSampler.from_worker_factory(workers, policy, env)
    rollouts = sampler.obtain_samples(0, 160, policy)
    assert sum(rollouts.lengths) >= 160
