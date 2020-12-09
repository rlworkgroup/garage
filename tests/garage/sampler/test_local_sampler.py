import numpy as np
import pytest

from garage.envs import GymEnv, PointEnv
from garage.envs.dm_control import DMControlEnv
from garage.experiment.task_sampler import SetTaskSampler
from garage.np.policies import FixedPolicy, ScriptedPolicy
from garage.sampler import LocalSampler, WorkerFactory


def test_update_envs_env_update():
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
    sampler = LocalSampler.from_worker_factory(workers, policy, env)
    episodes = sampler.obtain_samples(0,
                                      161,
                                      np.asarray(policy.get_param_values()),
                                      env_update=tasks.sample(n_workers))
    mean_rewards = []
    goals = []
    for eps in episodes.split():
        mean_rewards.append(eps.rewards.mean())
        goals.append(eps.env_infos['task'][0]['goal'])
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
    sampler = LocalSampler.from_worker_factory(workers,
                                               policy,
                                               envs=tasks.sample(n_workers))
    episodes = sampler.obtain_samples(0, 160, policy)
    assert sum(episodes.lengths) >= 160


def test_obtain_exact_episodes():
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
    sampler = LocalSampler.from_worker_factory(workers, policies, envs=env)
    n_eps_per_worker = 3
    episodes = sampler.obtain_exact_episodes(n_eps_per_worker,
                                             agent_update=policies)
    # At least one action per episode.
    assert sum(episodes.lengths) >= n_workers * n_eps_per_worker
    # All of the episodes.
    assert len(episodes.lengths) == n_workers * n_eps_per_worker
    worker = -1
    for count, eps in enumerate(episodes.split()):
        if count % n_eps_per_worker == 0:
            worker += 1
        assert (eps.actions == per_worker_actions[worker]).all()


def test_no_seed():
    max_episode_length = 16
    env = PointEnv()
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
    episodes = sampler.obtain_samples(0, 160, policy)
    assert sum(episodes.lengths) >= 160


def test_deterministic_on_policy_sampling_gym_env():
    max_episode_length = 10
    env1 = GymEnv('LunarLander-v2')
    env2 = GymEnv('LunarLander-v2')
    # Fix the action sequence
    env1.action_space.seed(10)
    env2.action_space.seed(10)
    policy1 = FixedPolicy(env1.spec,
                          scripted_actions=[
                              env1.action_space.sample()
                              for _ in range(max_episode_length)
                          ])
    policy2 = FixedPolicy(env2.spec,
                          scripted_actions=[
                              env2.action_space.sample()
                              for _ in range(max_episode_length)
                          ])
    n_workers = 2
    worker1 = WorkerFactory(seed=10,
                            max_episode_length=max_episode_length,
                            n_workers=n_workers)
    worker2 = WorkerFactory(seed=10,
                            max_episode_length=max_episode_length,
                            n_workers=n_workers)
    sampler1 = LocalSampler.from_worker_factory(worker1, policy1, env1)
    sampler2 = LocalSampler.from_worker_factory(worker2, policy2, env2)
    episodes1 = sampler1.obtain_samples(0, 10, policy1)
    episodes2 = sampler2.obtain_samples(0, 10, policy2)
    assert np.array_equal(episodes1.observations, episodes2.observations)
    assert np.array_equal(episodes1.next_observations,
                          episodes2.next_observations)


def test_deterministic_on_policy_sampling_dm_env():
    max_episode_length = 10
    env1 = DMControlEnv.from_suite('cartpole', 'balance')
    env2 = DMControlEnv.from_suite('cartpole', 'balance')
    # Fix the action sequence
    env1.action_space.seed(10)
    env2.action_space.seed(10)
    policy1 = FixedPolicy(env1.spec,
                          scripted_actions=[
                              env1.action_space.sample()
                              for _ in range(max_episode_length)
                          ])
    policy2 = FixedPolicy(env2.spec,
                          scripted_actions=[
                              env2.action_space.sample()
                              for _ in range(max_episode_length)
                          ])
    n_workers = 2
    worker1 = WorkerFactory(seed=10,
                            max_episode_length=max_episode_length,
                            n_workers=n_workers)
    worker2 = WorkerFactory(seed=10,
                            max_episode_length=max_episode_length,
                            n_workers=n_workers)
    sampler1 = LocalSampler.from_worker_factory(worker1, policy1, env1)
    sampler2 = LocalSampler.from_worker_factory(worker2, policy2, env2)
    episodes1 = sampler1.obtain_samples(0, 10, policy1)
    episodes2 = sampler2.obtain_samples(0, 10, policy2)
    assert np.array_equal(episodes1.observations, episodes2.observations)
    assert np.array_equal(episodes1.next_observations,
                          episodes2.next_observations)


if __name__ == '__main__':
    test_deterministic_on_policy_sampling_dm_env()
