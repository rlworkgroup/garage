import numpy as np

from garage.replay_buffer import PathBuffer
from tests.fixtures.envs.dummy import DummyDiscreteEnv


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

        # Can still sample from old path
        new_sampled_obs = replay_buffer.sample_transitions(1000)['obs']
        assert set(new_sampled_obs.flatten()) == {1, 2, 3}

        # Can't sample complete old path
        for _ in range(100):
            new_sampled_path_obs = replay_buffer.sample_path()['obs']
            assert (new_sampled_path_obs == np.array([[2], [3]])).all()
