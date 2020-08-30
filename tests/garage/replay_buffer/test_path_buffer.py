# pylint: disable=protected-access
import akro
import numpy as np
import pytest

from garage import EnvSpec, EpisodeBatch, StepType
from garage.replay_buffer import PathBuffer

from tests.fixtures.envs.dummy import DummyDiscreteEnv


@pytest.fixture
def eps_data():
    # spaces
    obs_space = akro.Box(low=1, high=np.inf, shape=(4, 3, 2), dtype=np.float32)
    act_space = akro.Discrete(2)
    env_spec = EnvSpec(obs_space, act_space)

    # generate data
    lens = np.array([10, 20, 7, 25, 25, 40, 10, 5])
    n_t = lens.sum()
    obs = np.stack([obs_space.low] * n_t)
    last_obs = np.stack([obs_space.low] * len(lens))
    act = np.stack([1] * n_t)
    rew = np.arange(n_t)

    # env_infos
    env_infos = dict()
    env_infos['goal'] = np.stack([[1, 1]] * n_t)
    env_infos['foo'] = np.arange(n_t)

    # agent_infos
    agent_infos = dict()
    agent_infos['prev_action'] = act
    agent_infos['hidden'] = np.arange(n_t)

    # step_types
    step_types = []
    for size in lens:
        step_types.extend([StepType.FIRST] + [StepType.MID] * (size - 2) +
                          [StepType.TERMINAL])
    step_types = np.array(step_types, dtype=StepType)

    # episode_infos
    episode_infos = dict()
    episode_infos['task_one_hot'] = np.stack([[1, 1]] * len(lens))

    return {
        'env_spec': env_spec,
        'episode_infos': episode_infos,
        'observations': obs,
        'last_observations': last_obs,
        'actions': act,
        'rewards': rew,
        'env_infos': env_infos,
        'agent_infos': agent_infos,
        'step_types': step_types,
        'lengths': lens
    }


class TestPathBuffer:

    def test_add_path_dtype(self):
        env = DummyDiscreteEnv()
        obs = env.reset()
        replay_buffer = PathBuffer(capacity_in_transitions=3)
        replay_buffer.add_path({
            'observations':
            np.array([obs]),
            'actions':
            np.array([[env.action_space.sample()]])
        })
        sample = replay_buffer.sample_transitions(1)
        sample_obs = sample['observations']
        sample_action = sample['actions']

        assert sample_obs.dtype == env.observation_space.dtype
        assert sample_action.dtype == env.action_space.dtype

    def test_episode_batch_to_timestep_batch(self, eps_data):
        t = EpisodeBatch(**eps_data)
        replay_buffer = PathBuffer(capacity_in_transitions=100)
        replay_buffer.add_episode_batch(t)
        timesteps = replay_buffer.sample_timesteps(10)
        assert len(timesteps.rewards) == 10

    def test_eviction_policy(self):
        obs = np.array([[1], [1]])
        replay_buffer = PathBuffer(capacity_in_transitions=3)
        replay_buffer.add_path(dict(obs=obs))

        sampled_obs = replay_buffer.sample_transitions(3)['obs']
        assert (sampled_obs == np.array([[1], [1], [1]])).all()

        sampled_path_obs = replay_buffer.sample_path()['obs']
        assert (sampled_path_obs == np.array([[1], [1]])).all()

        obs2 = np.array([[2], [3]])
        replay_buffer.add_path(dict(obs=obs2))

        with pytest.raises(Exception):
            assert replay_buffer.add_path(dict(test_obs=obs2))

        obs3 = np.array([1])
        with pytest.raises(Exception):
            assert replay_buffer.add_path(dict(obs=obs3))

        obs4 = np.array([[4], [5], [6], [7]])
        with pytest.raises(Exception):
            assert replay_buffer.add_path(dict(obs=obs4))

        # Can still sample from old path
        new_sampled_obs = replay_buffer.sample_transitions(1000)['obs']
        assert set(new_sampled_obs.flatten()) == {1, 2, 3}

        # Can't sample complete old path
        for _ in range(100):
            new_sampled_path_obs = replay_buffer.sample_path()['obs']
            assert (new_sampled_path_obs == np.array([[2], [3]])).all()

        replay_buffer.clear()
        assert replay_buffer.n_transitions_stored == 0
        assert not replay_buffer._buffer
