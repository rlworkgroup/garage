import pprint

import pytest

from garage.envs import GarageEnv
from garage.envs import GridWorldEnv
from garage.experiment.task_sampler import EnvPoolSampler
from garage.np.policies import ScriptedPolicy
from garage.sampler import LocalSampler, VecWorker, WorkerFactory

SEED = 100
N_TRAJ = 5
MAX_PATH_LENGTH = 9


@pytest.fixture
def env():
    return GarageEnv(GridWorldEnv(desc='4x4'))


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
    return [GarageEnv(GridWorldEnv(desc=desc)) for desc in descs]


@pytest.fixture
def other_envs():
    descs = [
        ['FFFS', 'FHFH', 'FFFH', 'HFFG'],
        ['FFSF', 'FFFH', 'FHFH', 'HFFG'],
        ['FFFF', 'FFSH', 'FHFH', 'FFFG'],
        ['FFFF', 'FFFF', 'FSFF', 'FFFF'],
        ['HHFF', 'HHHF', 'HSHF', 'HHHF'],
    ]
    return [GarageEnv(GridWorldEnv(desc=desc)) for desc in descs]


def assert_trajs_eq(ground_truth_traj, test_traj):
    # We should have the exact same trajectories.
    ground_truth_set = {(tuple(traj.actions), tuple(traj.observations))
                        for traj in ground_truth_traj.split()}
    test_set = {(tuple(traj.actions), tuple(traj.observations))
                for traj in test_traj.split()}
    print()
    pprint.pprint(ground_truth_set)
    pprint.pprint(test_set)
    assert test_set == ground_truth_set


def test_rollout(env, policy):
    worker = VecWorker(seed=SEED,
                       max_path_length=MAX_PATH_LENGTH,
                       worker_number=0,
                       n_envs=N_TRAJ)
    worker.update_agent(policy)
    worker.update_env(env)
    traj = worker.rollout()
    assert len(traj.lengths) == N_TRAJ
    traj2 = worker.rollout()
    assert len(traj2.lengths) == N_TRAJ
    assert str(traj) == str(traj2)
    assert traj.actions.var() > 0
    worker.shutdown()


def test_non_vec_rollout(env, policy):
    worker = VecWorker(seed=SEED,
                       max_path_length=MAX_PATH_LENGTH,
                       worker_number=0,
                       n_envs=1)
    worker.update_agent(policy)
    worker.update_env(env)
    traj = worker.rollout()
    assert len(traj.lengths) == 1
    assert traj.actions.var() > 0
    traj2 = worker.rollout()
    assert len(traj2.lengths) == 1
    worker.shutdown()


def test_in_local_sampler(policy, envs):
    true_workers = WorkerFactory(seed=100,
                                 n_workers=N_TRAJ,
                                 max_path_length=MAX_PATH_LENGTH)
    true_sampler = LocalSampler.from_worker_factory(true_workers, policy, envs)
    vec_workers = WorkerFactory(seed=100,
                                n_workers=1,
                                worker_class=VecWorker,
                                worker_args=dict(n_envs=N_TRAJ),
                                max_path_length=MAX_PATH_LENGTH)
    vec_sampler = LocalSampler.from_worker_factory(vec_workers, policy, [envs])
    n_samples = 100

    true_trajs = true_sampler.obtain_samples(0, n_samples, None)
    vec_trajs = vec_sampler.obtain_samples(0, n_samples, None)
    assert vec_trajs.lengths.sum() >= n_samples
    assert_trajs_eq(true_trajs, vec_trajs)

    # Test start_rollout optimization

    true_trajs = true_sampler.obtain_samples(0, n_samples, None)
    vec_trajs = vec_sampler.obtain_samples(0, n_samples, None)
    assert vec_trajs.lengths.sum() >= n_samples
    assert_trajs_eq(true_trajs, vec_trajs)

    true_sampler.shutdown_worker()
    vec_sampler.shutdown_worker()


def test_reset_optimization(policy, envs, other_envs):
    true_workers = WorkerFactory(seed=100,
                                 n_workers=N_TRAJ,
                                 max_path_length=MAX_PATH_LENGTH)
    true_sampler = LocalSampler.from_worker_factory(true_workers, policy, envs)
    vec_workers = WorkerFactory(seed=100,
                                n_workers=1,
                                worker_class=VecWorker,
                                worker_args=dict(n_envs=N_TRAJ),
                                max_path_length=MAX_PATH_LENGTH)
    vec_sampler = LocalSampler.from_worker_factory(vec_workers, [policy],
                                                   [envs])
    n_samples = 4 * MAX_PATH_LENGTH
    true_sampler.obtain_samples(0, n_samples, None)
    true_sampler.obtain_samples(0, n_samples, None)

    true_trajs = true_sampler.obtain_samples(0, n_samples, None, other_envs)
    vec_trajs = vec_sampler.obtain_samples(0, n_samples, None, [other_envs])

    assert vec_trajs.lengths.sum() >= n_samples
    assert_trajs_eq(true_trajs, vec_trajs)

    true_sampler.shutdown_worker()
    vec_sampler.shutdown_worker()


def test_init_with_env_updates(policy, envs):
    task_sampler = EnvPoolSampler(envs)
    envs = task_sampler.sample(N_TRAJ)
    true_workers = WorkerFactory(seed=100,
                                 n_workers=N_TRAJ,
                                 max_path_length=MAX_PATH_LENGTH)
    true_sampler = LocalSampler.from_worker_factory(true_workers, policy, envs)
    vec_workers = WorkerFactory(seed=100,
                                n_workers=1,
                                worker_class=VecWorker,
                                worker_args=dict(n_envs=N_TRAJ),
                                max_path_length=MAX_PATH_LENGTH)
    vec_sampler = LocalSampler.from_worker_factory(vec_workers, [policy],
                                                   [envs])
    n_samples = 100
    true_trajs = true_sampler.obtain_samples(0, n_samples, None)
    vec_trajs = vec_sampler.obtain_samples(0, n_samples, None)

    assert vec_trajs.lengths.sum() >= n_samples
    assert_trajs_eq(true_trajs, vec_trajs)

    true_sampler.shutdown_worker()
    vec_sampler.shutdown_worker()
