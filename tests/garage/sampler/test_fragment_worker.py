import math
import pprint

import numpy as np
import pytest

from garage import TrajectoryBatch
from garage.envs import GarageEnv, GridWorldEnv
from garage.np.policies import ScriptedPolicy
from garage.sampler import FragmentWorker, LocalSampler, WorkerFactory

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
        ['SFFF', 'FFFF', 'FFFF', 'FFFF'],
        ['FFSF', 'FFFH', 'FHFH', 'HFFG'],
        ['FHSF', 'FFFH', 'FHFH', 'HFFG'],
        ['FHSF', 'FGFH', 'FHFH', 'HFFH'],
        ['SHFF', 'HHFF', 'FFFF', 'FFFF'],
    ]
    return [GarageEnv(GridWorldEnv(desc=desc)) for desc in descs]


def trajs_eq(true_traj, test_traj):
    return (np.allclose(true_traj.observations, test_traj.observations)
            and np.allclose(true_traj.actions, test_traj.actions))


def slice_trajectories(trajectories, slice_size):
    sliced = []
    for traj in trajectories.split():
        splits = math.ceil(traj.lengths[0] / slice_size)
        split_indices = np.array_split(np.arange(traj.lengths[0]), splits)
        next_obs = traj.next_observations
        for indices in split_indices:
            last_obs = np.asarray([next_obs[indices[-1]]])
            t = TrajectoryBatch(
                env_spec=traj.env_spec,
                observations=traj.observations[indices],
                last_observations=last_obs,
                actions=traj.actions[indices],
                rewards=traj.rewards[indices],
                terminals=traj.terminals[indices],
                env_infos={k: v[indices]
                           for (k, v) in traj.env_infos},
                agent_infos={k: v[indices]
                             for (k, v) in traj.agent_infos},
                lengths=np.asarray([len(indices)], dtype='l'))
            sliced.append(t)
    return sliced


@pytest.mark.parametrize('timesteps_per_call', [1, 2])
def test_rollout(env, policy, timesteps_per_call):
    worker = FragmentWorker(seed=SEED,
                            max_path_length=MAX_PATH_LENGTH,
                            worker_number=0,
                            n_envs=N_TRAJ,
                            timesteps_per_call=timesteps_per_call)
    worker.update_agent(policy)
    worker.update_env(env)
    n_calls = math.ceil(MAX_PATH_LENGTH / timesteps_per_call)
    for i in range(n_calls):
        traj = worker.rollout()
        assert sum(traj.lengths) == timesteps_per_call * N_TRAJ
        if timesteps_per_call * i < 4:
            assert sum(traj.terminals) == 0
    worker.shutdown()


@pytest.mark.parametrize('timesteps_per_call', [1, 2])
def test_in_local_sampler(policy, envs, other_envs, timesteps_per_call):
    true_workers = WorkerFactory(seed=100,
                                 n_workers=N_TRAJ,
                                 max_path_length=MAX_PATH_LENGTH)
    true_sampler = LocalSampler.from_worker_factory(true_workers, policy, envs)
    worker_args = dict(n_envs=N_TRAJ, timesteps_per_call=timesteps_per_call)
    vec_workers = WorkerFactory(seed=100,
                                n_workers=1,
                                worker_class=FragmentWorker,
                                worker_args=worker_args,
                                max_path_length=MAX_PATH_LENGTH)
    vec_sampler = LocalSampler.from_worker_factory(vec_workers, policy, [envs])
    n_samples = 400

    true_trajs = true_sampler.obtain_samples(0, n_samples, None)
    sliced_true_trajs = slice_trajectories(true_trajs, timesteps_per_call)

    vec_trajs = vec_sampler.obtain_samples(0, 50, None)
    for test_traj in vec_trajs.split():
        assert any(
            trajs_eq(true_traj, test_traj) for true_traj in sliced_true_trajs)

    true_trajs = true_sampler.obtain_samples(0,
                                             n_samples,
                                             None,
                                             env_update=other_envs)
    sliced_true_trajs = slice_trajectories(true_trajs, timesteps_per_call)

    vec_trajs = vec_sampler.obtain_samples(0,
                                           50,
                                           None,
                                           env_update=[other_envs])
    for test_traj in vec_trajs.split():
        assert any(
            trajs_eq(true_traj, test_traj) for true_traj in sliced_true_trajs)

    true_sampler.shutdown_worker()
    vec_sampler.shutdown_worker()
