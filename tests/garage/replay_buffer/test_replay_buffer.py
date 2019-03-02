import unittest

from garage.replay_buffer import SimpleReplayBuffer
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class TestReplayBuffer(unittest.TestCase):
    def test_replay_buffer_dtype(self):
        env = DummyDiscreteEnv()
        obs = env.reset()
        replay_buffer = SimpleReplayBuffer(
            env_spec=env, size_in_transitions=100, time_horizon=1)
        replay_buffer.add_transition(
            observation=[obs], action=[env.action_space.sample()])
        sample = replay_buffer.sample(1)
        sample_obs = sample['observation']
        sample_action = sample['action']

        assert sample_obs.dtype == env.observation_space.dtype
        assert sample_action.dtype == env.action_space.dtype
