from unittest.mock import Mock

import numpy as np
import pytest

from garage.envs import GarageEnv
from garage.envs import PointEnv
from garage.envs.grid_world_env import GridWorldEnv
from garage.experiment.task_sampler import SetTaskSampler
from garage.np.policies import FixedPolicy, ScriptedPolicy
from garage.sampler import LocalSampler, OnPolicyVectorizedSampler
from garage.sampler import WorkerFactory


class TestSampler:
    """
    Uses mock policy for 4x4 gridworldenv
    '4x4': [
        'SFFF',
        'FHFH',
        'FFFH',
        'HFFG'
    ]
    0: left
    1: down
    2: right
    3: up
    -1: no move
    'S' : starting point
    'F' or '.': free space
    'W' or 'x': wall
    'H' or 'o': hole (terminates episode)
    'G' : goal
    [2,2,1,0,3,1,1,1,2,2,1,1,1,2,2,1]
    """

    def setup_method(self):
        self.env = GarageEnv(GridWorldEnv(desc='4x4'))
        self.policy = ScriptedPolicy(
            scripted_actions=[2, 2, 1, 0, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1])
        self.algo = Mock(env_spec=self.env.spec,
                         policy=self.policy,
                         max_path_length=16)

    def teardown_method(self):
        self.env.close()

    def test_local_batch_sampler(self):
        workers = WorkerFactory(seed=100,
                                max_path_length=self.algo.max_path_length)
        sampler1 = LocalSampler.from_worker_factory(workers, self.policy,
                                                    self.env)
        sampler2 = OnPolicyVectorizedSampler(self.algo, self.env)
        sampler2.start_worker()
        trajs1 = sampler1.obtain_samples(
            0, 1000, tuple(self.algo.policy.get_param_values()))
        trajs2 = sampler2.obtain_samples(0, 1000)
        # pylint: disable=superfluous-parens
        assert trajs1.observations.shape[0] >= 1000
        assert trajs1.actions.shape[0] >= 1000
        assert (sum(trajs1.rewards[:trajs1.lengths[0]]) == sum(
            trajs2[0]['rewards']) == 1)

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
    sampler = LocalSampler.from_worker_factory(workers,
                                               policy,
                                               envs=tasks.sample(n_workers))
    rollouts = sampler.obtain_samples(0, 160, policy)
    assert sum(rollouts.lengths) >= 160


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
