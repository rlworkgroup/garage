import pickle

import numpy as np

from garage.replay_buffer import SimpleReplayBuffer
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class TestReplayBuffer:

    def test_add_transition_dtype(self):
        env = DummyDiscreteEnv()
        obs = env.reset()
        replay_buffer = SimpleReplayBuffer(env_spec=env,
                                           size_in_transitions=3,
                                           time_horizon=1)
        replay_buffer.add_transition(observation=obs,
                                     action=env.action_space.sample())
        sample = replay_buffer.sample(1)
        sample_obs = sample['observation']
        sample_action = sample['action']

        assert sample_obs.dtype == env.observation_space.dtype
        assert sample_action.dtype == env.action_space.dtype

    def test_add_transitions_dtype(self):
        env = DummyDiscreteEnv()
        obs = env.reset()
        replay_buffer = SimpleReplayBuffer(env_spec=env,
                                           size_in_transitions=3,
                                           time_horizon=1)
        replay_buffer.add_transitions(observation=[obs],
                                      action=[env.action_space.sample()])
        sample = replay_buffer.sample(1)
        sample_obs = sample['observation']
        sample_action = sample['action']

        assert sample_obs.dtype == env.observation_space.dtype
        assert sample_action.dtype == env.action_space.dtype

    def test_eviction_policy(self):
        env = DummyDiscreteEnv()
        obs = env.reset()

        replay_buffer = SimpleReplayBuffer(env_spec=env,
                                           size_in_transitions=3,
                                           time_horizon=1)
        replay_buffer.add_transitions(observation=[obs, obs], action=[1, 2])
        assert not replay_buffer.full
        replay_buffer.add_transitions(observation=[obs, obs], action=[3, 4])
        assert replay_buffer.full
        replay_buffer.add_transitions(observation=[obs, obs], action=[5, 6])
        replay_buffer.add_transitions(observation=[obs, obs], action=[7, 8])

        assert np.array_equal(replay_buffer._buffer['action'], [[7], [8], [6]])
        assert replay_buffer.n_transitions_stored == 3

    def test_pickleable(self):
        env = DummyDiscreteEnv()
        obs = env.reset()

        replay_buffer = SimpleReplayBuffer(env_spec=env,
                                           size_in_transitions=100,
                                           time_horizon=1)
        for _ in range(0, 100):
            replay_buffer.add_transitions(observation=[obs], action=[1])
        replay_buffer_pickled = pickle.loads(pickle.dumps(replay_buffer))
        assert replay_buffer_pickled._buffer.keys(
        ) == replay_buffer._buffer.keys()
        for k in replay_buffer_pickled._buffer:
            assert replay_buffer_pickled._buffer[
                k].shape == replay_buffer._buffer[k].shape
