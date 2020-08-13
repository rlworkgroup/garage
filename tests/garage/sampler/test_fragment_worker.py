# pylint is confused by pytest.mark.parametrize
# pylint: disable=not-callable
import math
import pprint

import numpy as np
import pytest

from garage import EpisodeBatch, StepType
from garage.envs import GridWorldEnv
from garage.np.policies import ScriptedPolicy
from garage.sampler import FragmentWorker, LocalSampler, WorkerFactory

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
        ['SFFF', 'FFFF', 'FFFF', 'FFFF'],
        ['FFSF', 'FFFH', 'FHFH', 'HFFG'],
        ['FHSF', 'FFFH', 'FHFH', 'HFFG'],
        ['FHSF', 'FGFH', 'FHFH', 'HFFH'],
        ['SHFF', 'HHFF', 'FFFF', 'FFFF'],
    ]
    return [GridWorldEnv(desc=desc) for desc in descs]


def eps_eq(true_eps, test_eps):
    return (np.allclose(true_eps.observations, test_eps.observations)
            and np.allclose(true_eps.actions, test_eps.actions))


def slice_episodes(episodes, slice_size):
    sliced = []
    for eps in episodes.split():
        splits = math.ceil(eps.lengths[0] / slice_size)
        split_indices = np.array_split(np.arange(eps.lengths[0]), splits)
        next_obs = eps.next_observations
        for indices in split_indices:
            last_obs = np.asarray([next_obs[indices[-1]]])
            t = EpisodeBatch(
                env_spec=eps.env_spec,
                observations=eps.observations[indices],
                last_observations=last_obs,
                actions=eps.actions[indices],
                rewards=eps.rewards[indices],
                step_types=eps.step_types[indices],
                env_infos={k: v[indices]
                           for (k, v) in eps.env_infos.items()},
                agent_infos={
                    k: v[indices]
                    for (k, v) in eps.agent_infos.items()
                },
                lengths=np.asarray([len(indices)], dtype='l'))
            sliced.append(t)
    return sliced


# pylint: disable=not-callable
@pytest.mark.parametrize('timesteps_per_call', [1, 2])
def test_rollout(env, policy, timesteps_per_call):
    worker = FragmentWorker(seed=SEED,
                            max_episode_length=MAX_EPISODE_LENGTH,
                            worker_number=0,
                            n_envs=N_EPS,
                            timesteps_per_call=timesteps_per_call)
    worker.update_agent(policy)
    worker.update_env(env)
    n_calls = math.ceil(MAX_EPISODE_LENGTH / timesteps_per_call)
    for i in range(n_calls):
        eps = worker.rollout()
        assert sum(eps.lengths) == timesteps_per_call * N_EPS
        if timesteps_per_call * i < 4:
            assert not any(step_type == StepType.TERMINAL
                           for step_type in eps.step_types)
    worker.shutdown()


# pylint: disable=not-callable
@pytest.mark.parametrize('timesteps_per_call', [1, 2])
def test_in_local_sampler(policy, envs, other_envs, timesteps_per_call):
    true_workers = WorkerFactory(seed=100,
                                 n_workers=N_EPS,
                                 max_episode_length=MAX_EPISODE_LENGTH)
    true_sampler = LocalSampler.from_worker_factory(true_workers, policy, envs)
    worker_args = dict(n_envs=N_EPS, timesteps_per_call=timesteps_per_call)
    vec_workers = WorkerFactory(seed=100,
                                n_workers=1,
                                worker_class=FragmentWorker,
                                worker_args=worker_args,
                                max_episode_length=MAX_EPISODE_LENGTH)
    vec_sampler = LocalSampler.from_worker_factory(vec_workers, policy, [envs])
    n_samples = 400

    true_eps = true_sampler.obtain_samples(0, n_samples, None)
    sliced_true_eps = slice_episodes(true_eps, timesteps_per_call)

    vec_eps = vec_sampler.obtain_samples(0, 50, None)
    for test_eps in vec_eps.split():
        assert any(eps_eq(true_eps, test_eps) for true_eps in sliced_true_eps)

    true_eps = true_sampler.obtain_samples(0,
                                           n_samples,
                                           None,
                                           env_update=other_envs)
    sliced_true_eps = slice_episodes(true_eps, timesteps_per_call)

    vec_eps = vec_sampler.obtain_samples(0, 50, None, env_update=[other_envs])
    for test_eps in vec_eps.split():
        assert any(eps_eq(true_eps, test_eps) for true_eps in sliced_true_eps)

    true_sampler.shutdown_worker()
    vec_sampler.shutdown_worker()
