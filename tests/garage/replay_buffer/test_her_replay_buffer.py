import pickle

import numpy as np
import pytest

from garage.replay_buffer import HerReplayBuffer
from tests.fixtures.envs.dummy import DummyDictEnv


class TestHerReplayBuffer:

    def setup_method(self):
        self.env = DummyDictEnv()
        self.obs = self.env.reset()
        self._replay_k = 4
        self.replay_buffer = HerReplayBuffer(
            env_spec=self.env.spec,
            capacity_in_transitions=10,
            replay_k=self._replay_k,
            reward_fun=self.env.compute_reward)

    def test_replay_k(self):
        self.replay_buffer = HerReplayBuffer(
            env_spec=self.env.spec,
            capacity_in_transitions=10,
            replay_k=0,
            reward_fun=self.env.compute_reward)

        with pytest.raises(ValueError):
            self.replay_buffer = HerReplayBuffer(
                env_spec=self.env.spec,
                capacity_in_transitions=10,
                replay_k=0.2,
                reward_fun=self.env.compute_reward)

    def _add_one_path(self):
        path = dict(
            observations=np.asarray([self.obs, self.obs]),
            actions=np.asarray([[self.env.action_space.sample()],
                                [self.env.action_space.sample()]]),
            rewards=np.asarray([[1], [1]]),
            terminals=np.asarray([[False], [False]]),
            next_observations=np.asarray([self.obs, self.obs]),
        )
        self.replay_buffer.add_path(path)

    def test_add_path(self):
        self._add_one_path()

        # HER buffer should add replay_k + 1 transitions to the buffer
        # for each transition in the given path. This doesn't apply to
        # the last transition, where only that transition gets added.

        path_len = 2
        total_expected_transitions = sum(
            [self._replay_k + 1 for _ in range(path_len - 1)]) + 1
        assert (self.replay_buffer.n_transitions_stored ==
                total_expected_transitions)
        assert len(self.replay_buffer._path_segments) == 1

        # check that buffer has the correct keys
        assert {
            'observations', 'achieved_goals', 'desired_goals',
            'next_observations', 'next_achieved_goals', 'next_desired_goals'
        } <= set(self.replay_buffer._buffer)

    def test_sample_transitions(self):
        self._add_one_path()

        transitions = self.replay_buffer.sample_transitions(3)

        # make sure sample observations are flattened
        for obs_type in ['observations', 'next_observations']:
            for obs in transitions[obs_type]:
                flat_obs = self.env.observation_space.flatten(self.obs)
                assert len(obs) == len(flat_obs)

        # check that transitions have the right keys
        assert {'rewards', 'terminals', 'actions'} <= set(transitions)

    def test_pickleable(self):
        self._add_one_path()
        replay_buffer_pickled = pickle.loads(pickle.dumps(self.replay_buffer))
        assert (replay_buffer_pickled._buffer.keys() ==
                self.replay_buffer._buffer.keys())
        for k in replay_buffer_pickled._buffer:
            assert replay_buffer_pickled._buffer[
                k].shape == self.replay_buffer._buffer[k].shape
        sample = self.replay_buffer.sample_transitions(1)
        sample2 = replay_buffer_pickled.sample_transitions(1)
        for k in sample.keys():
            assert sample[k].shape == sample2[k].shape
        assert len(sample) == len(sample2)
