import pickle
from unittest.mock import Mock

import numpy as np
import pytest

from garage.envs import GarageEnv
from garage.envs import PointEnv
from garage.envs.grid_world_env import GridWorldEnv
from garage.experiment.task_sampler import SetTaskSampler
from garage.np.policies import FixedPolicy, ScriptedPolicy
from garage.sampler import LocalSampler, MultiprocessingSampler
from garage.sampler import WorkerFactory


@pytest.mark.timeout(10)
def test_obtain_samples():
    env = GarageEnv(GridWorldEnv(desc='4x4'))
    policy = ScriptedPolicy(
        scripted_actions=[2, 2, 1, 0, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1])
    algo = Mock(env_spec=env.spec, policy=policy, max_path_length=16)

    workers = WorkerFactory(seed=100,
                            max_path_length=algo.max_path_length,
                            n_workers=8)
    sampler1 = MultiprocessingSampler.from_worker_factory(workers, policy, env)
    sampler2 = LocalSampler.from_worker_factory(workers, policy, env)
    trajs1 = sampler1.obtain_samples(0, 1000,
                                     tuple(algo.policy.get_param_values()))
    trajs2 = sampler2.obtain_samples(0, 1000,
                                     tuple(algo.policy.get_param_values()))
    # pylint: disable=superfluous-parens
    assert trajs1.observations.shape[0] >= 1000
    assert trajs1.actions.shape[0] >= 1000
    assert (sum(trajs1.rewards[:trajs1.lengths[0]]) == sum(
        trajs2.rewards[:trajs2.lengths[0]]) == 1)

    true_obs = np.array([0, 1, 2, 6, 10, 14])
    true_actions = np.array([2, 2, 1, 1, 1, 2])
    true_rewards = np.array([0, 0, 0, 0, 0, 1])
    start = 0
    for length in trajs1.lengths:
        observations = trajs1.observations[start:start + length]
        actions = trajs1.actions[start:start + length]
        rewards = trajs1.rewards[start:start + length]
        assert np.array_equal(observations, true_obs)
        assert np.array_equal(actions, true_actions)
        assert np.array_equal(rewards, true_rewards)
        start += length
    sampler1.shutdown_worker()
    sampler2.shutdown_worker()
    env.close()


@pytest.mark.timeout(10)
def test_update_envs_env_update():
    max_path_length = 16
    env = GarageEnv(PointEnv())
    policy = FixedPolicy(env.spec,
                         scripted_actions=[
                             env.action_space.sample()
                             for _ in range(max_path_length)
                         ])
    tasks = SetTaskSampler(PointEnv)
    n_workers = 8
    workers = WorkerFactory(seed=100,
                            max_path_length=max_path_length,
                            n_workers=n_workers)
    sampler = MultiprocessingSampler.from_worker_factory(workers, policy, env)
    rollouts = sampler.obtain_samples(0,
                                      161,
                                      np.asarray(policy.get_param_values()),
                                      env_update=tasks.sample(n_workers))
    mean_rewards = []
    goals = []
    for rollout in rollouts.split():
        mean_rewards.append(rollout.rewards.mean())
        goals.append(rollout.env_infos['task'][0]['goal'])
    assert np.var(mean_rewards) > 0
    assert np.var(goals) > 0
    with pytest.raises(ValueError):
        sampler.obtain_samples(0,
                               10,
                               np.asarray(policy.get_param_values()),
                               env_update=tasks.sample(n_workers + 1))
    sampler.shutdown_worker()
    env.close()


@pytest.mark.timeout(10)
def test_init_with_env_updates():
    max_path_length = 16
    env = GarageEnv(PointEnv())
    policy = FixedPolicy(env.spec,
                         scripted_actions=[
                             env.action_space.sample()
                             for _ in range(max_path_length)
                         ])
    tasks = SetTaskSampler(lambda: GarageEnv(PointEnv()))
    n_workers = 8
    workers = WorkerFactory(seed=100,
                            max_path_length=max_path_length,
                            n_workers=n_workers)
    sampler = MultiprocessingSampler.from_worker_factory(
        workers, policy, envs=tasks.sample(n_workers))
    rollouts = sampler.obtain_samples(0, 160, policy)
    assert sum(rollouts.lengths) >= 160
    sampler.shutdown_worker()
    env.close()


@pytest.mark.timeout(10)
def test_obtain_exact_trajectories():
    max_path_length = 15
    n_workers = 8
    env = GarageEnv(PointEnv())
    per_worker_actions = [env.action_space.sample() for _ in range(n_workers)]
    policies = [
        FixedPolicy(env.spec, [action] * max_path_length)
        for action in per_worker_actions
    ]
    workers = WorkerFactory(seed=100,
                            max_path_length=max_path_length,
                            n_workers=n_workers)
    sampler = MultiprocessingSampler.from_worker_factory(workers,
                                                         policies,
                                                         envs=env)
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
    sampler.shutdown_worker()
    env.close()


@pytest.mark.timeout(30)
def test_init_with_crashed_worker():
    max_path_length = 16
    env = GarageEnv(PointEnv())
    policy = FixedPolicy(env.spec,
                         scripted_actions=[
                             env.action_space.sample()
                             for _ in range(max_path_length)
                         ])
    tasks = SetTaskSampler(lambda: GarageEnv(PointEnv()))
    n_workers = 2
    workers = WorkerFactory(seed=100,
                            max_path_length=max_path_length,
                            n_workers=n_workers)

    class CrashingPolicy:

        def reset(self, **kwargs):
            raise Exception('Intentional subprocess crash')

    bad_policy = CrashingPolicy()

    #  This causes worker 2 to crash.
    sampler = MultiprocessingSampler.from_worker_factory(
        workers, [policy, bad_policy], envs=tasks.sample(n_workers))
    rollouts = sampler.obtain_samples(0, 160, None)
    assert sum(rollouts.lengths) >= 160
    sampler.shutdown_worker()
    env.close()


@pytest.mark.timeout(10)
def test_pickle():
    max_path_length = 16
    env = GarageEnv(PointEnv())
    policy = FixedPolicy(env.spec,
                         scripted_actions=[
                             env.action_space.sample()
                             for _ in range(max_path_length)
                         ])
    tasks = SetTaskSampler(PointEnv)
    n_workers = 8
    workers = WorkerFactory(seed=100,
                            max_path_length=max_path_length,
                            n_workers=n_workers)
    sampler = MultiprocessingSampler.from_worker_factory(workers, policy, env)
    sampler_pickled = pickle.dumps(sampler)
    sampler.shutdown_worker()
    sampler2 = pickle.loads(sampler_pickled)
    rollouts = sampler2.obtain_samples(0,
                                       161,
                                       np.asarray(policy.get_param_values()),
                                       env_update=tasks.sample(n_workers))
    mean_rewards = []
    goals = []
    for rollout in rollouts.split():
        mean_rewards.append(rollout.rewards.mean())
        goals.append(rollout.env_infos['task'][0]['goal'])
    assert np.var(mean_rewards) > 0
    assert np.var(goals) > 0
    sampler2.shutdown_worker()
    env.close()
