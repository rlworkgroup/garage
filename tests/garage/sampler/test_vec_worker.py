import pprint

import pytest

from garage.envs import GridWorldEnv
from garage.experiment.task_sampler import EnvPoolSampler
from garage.np.policies import ScriptedPolicy
from garage.sampler import LocalSampler, VecWorker, WorkerFactory

SEED = 100
N_EPS = 5
MAX_EPISODE_LENGTH = 9


@pytest.fixture
def env():
    return GridWorldEnv(desc='4x4')


@pytest.fixture
def policy():
    return ScriptedPolicy(
        scripted_actions=[2, 2, 1, 0, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1])


@pytest.fixture
def envs():
    descs = [
        ['SFFF', 'FHFH', 'FFFH', 'HFFG'],
        ['SFFF', 'FFFH', 'FHFH', 'HFFG'],
        ['SFFF', 'FFFH', 'FHFH', 'FFFG'],
        ['SFFF', 'FFFF', 'FFFF', 'FFFF'],
        ['SHFF', 'HHFF', 'FFFF', 'FFFF'],
    ]
    return [GridWorldEnv(desc=desc) for desc in descs]


@pytest.fixture
def other_envs():
    descs = [
        ['FFFS', 'FHFH', 'FFFH', 'HFFG'],
        ['FFSF', 'FFFH', 'FHFH', 'HFFG'],
        ['FFFF', 'FFSH', 'FHFH', 'FFFG'],
        ['FFFF', 'FFFF', 'FSFF', 'FFFF'],
        ['HHFF', 'HHHF', 'HSHF', 'HHHF'],
    ]
    return [GridWorldEnv(desc=desc) for desc in descs]


def assert_eps_eq(ground_truth_eps, test_eps):
    # We should have the exact same epsectories.
    ground_truth_set = {(tuple(eps.actions), tuple(eps.observations))
                        for eps in ground_truth_eps.split()}
    test_set = {(tuple(eps.actions), tuple(eps.observations))
                for eps in test_eps.split()}

    pprint.pprint(ground_truth_set)
    pprint.pprint(test_set)
    assert test_set == ground_truth_set


def test_rollout(env, policy):
    worker = VecWorker(seed=SEED,
                       max_episode_length=MAX_EPISODE_LENGTH,
                       worker_number=0,
                       n_envs=N_EPS)
    worker.update_agent(policy)
    worker.update_env(env)
    eps = worker.rollout()
    assert len(eps.lengths) == N_EPS
    eps2 = worker.rollout()
    assert len(eps2.lengths) == N_EPS
    assert str(eps) == str(eps2)
    assert eps.actions.var() > 0
    worker.shutdown()


def test_non_vec_rollout(env, policy):
    worker = VecWorker(seed=SEED,
                       max_episode_length=MAX_EPISODE_LENGTH,
                       worker_number=0,
                       n_envs=1)
    worker.update_agent(policy)
    worker.update_env(env)
    eps = worker.rollout()
    assert len(eps.lengths) == 1
    assert eps.actions.var() > 0
    eps2 = worker.rollout()
    assert len(eps2.lengths) == 1
    worker.shutdown()


def test_in_local_sampler(policy, envs):
    true_workers = WorkerFactory(seed=100,
                                 n_workers=N_EPS,
                                 max_episode_length=MAX_EPISODE_LENGTH)
    true_sampler = LocalSampler.from_worker_factory(true_workers, policy, envs)
    vec_workers = WorkerFactory(seed=100,
                                n_workers=1,
                                worker_class=VecWorker,
                                worker_args=dict(n_envs=N_EPS),
                                max_episode_length=MAX_EPISODE_LENGTH)
    vec_sampler = LocalSampler.from_worker_factory(vec_workers, policy, [envs])
    n_samples = 100

    true_eps = true_sampler.obtain_samples(0, n_samples, None)
    vec_eps = vec_sampler.obtain_samples(0, n_samples, None)
    assert vec_eps.lengths.sum() >= n_samples
    assert_eps_eq(true_eps, vec_eps)

    # Test start_rollout optimization

    true_eps = true_sampler.obtain_samples(0, n_samples, None)
    vec_eps = vec_sampler.obtain_samples(0, n_samples, None)
    assert vec_eps.lengths.sum() >= n_samples
    assert_eps_eq(true_eps, vec_eps)

    true_sampler.shutdown_worker()
    vec_sampler.shutdown_worker()


def test_reset_optimization(policy, envs, other_envs):
    true_workers = WorkerFactory(seed=100,
                                 n_workers=N_EPS,
                                 max_episode_length=MAX_EPISODE_LENGTH)
    true_sampler = LocalSampler.from_worker_factory(true_workers, policy, envs)
    vec_workers = WorkerFactory(seed=100,
                                n_workers=1,
                                worker_class=VecWorker,
                                worker_args=dict(n_envs=N_EPS),
                                max_episode_length=MAX_EPISODE_LENGTH)
    vec_sampler = LocalSampler.from_worker_factory(vec_workers, [policy],
                                                   [envs])
    n_samples = 4 * MAX_EPISODE_LENGTH
    true_sampler.obtain_samples(0, n_samples, None)
    true_sampler.obtain_samples(0, n_samples, None)

    true_eps = true_sampler.obtain_samples(0, n_samples, None, other_envs)
    vec_eps = vec_sampler.obtain_samples(0, n_samples, None, [other_envs])

    assert vec_eps.lengths.sum() >= n_samples
    assert_eps_eq(true_eps, vec_eps)

    true_sampler.shutdown_worker()
    vec_sampler.shutdown_worker()


def test_init_with_env_updates(policy, envs):
    task_sampler = EnvPoolSampler(envs)
    envs = task_sampler.sample(N_EPS)
    true_workers = WorkerFactory(seed=100,
                                 n_workers=N_EPS,
                                 max_episode_length=MAX_EPISODE_LENGTH)
    true_sampler = LocalSampler.from_worker_factory(true_workers, policy, envs)
    vec_workers = WorkerFactory(seed=100,
                                n_workers=1,
                                worker_class=VecWorker,
                                worker_args=dict(n_envs=N_EPS),
                                max_episode_length=MAX_EPISODE_LENGTH)
    vec_sampler = LocalSampler.from_worker_factory(vec_workers, [policy],
                                                   [envs])
    n_samples = 100
    true_eps = true_sampler.obtain_samples(0, n_samples, None)
    vec_eps = vec_sampler.obtain_samples(0, n_samples, None)

    assert vec_eps.lengths.sum() >= n_samples
    assert_eps_eq(true_eps, vec_eps)

    true_sampler.shutdown_worker()
    vec_sampler.shutdown_worker()
