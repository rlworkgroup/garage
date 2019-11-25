import pickle

import numpy as np

from garage.replay_buffer import HerReplayBuffer
from tests.fixtures.envs.dummy import DummyDictEnv


class TestHerReplayBuffer:

    def setup_method(self):
        self.env = DummyDictEnv()
        obs = self.env.reset()
        self.replay_buffer = HerReplayBuffer(
            env_spec=self.env.spec,
            size_in_transitions=3,
            time_horizon=1,
            replay_k=0.4,
            reward_fun=self.env.compute_reward)
        # process observations
        self.d_g = obs['desired_goal']
        self.a_g = obs['achieved_goal']
        self.obs = obs['observation']

    def _add_single_transition(self):
        self.replay_buffer.add_transition(
            observation=self.obs,
            action=self.env.action_space.sample(),
            goal=self.d_g,
            achieved_goal=self.a_g,
            next_observation=self.obs,
            next_achieved_goal=self.a_g)

    def _add_transitions(self):
        self.replay_buffer.add_transitions(
            observation=[self.obs],
            action=[self.env.action_space.sample()],
            goal=[self.d_g],
            achieved_goal=[self.a_g],
            next_observation=[self.obs],
            next_achieved_goal=[self.a_g])

    def test_add_transition_dtype(self):
        self._add_single_transition()
        sample = self.replay_buffer.sample(1)

        assert sample['observation'].dtype == self.env.observation_space[
            'observation'].dtype
        assert sample['achieved_goal'].dtype == self.env.observation_space[
            'achieved_goal'].dtype
        assert sample['goal'].dtype == self.env.observation_space[
            'desired_goal'].dtype
        assert sample['action'].dtype == self.env.action_space.dtype

    def test_add_transitions_dtype(self):
        self._add_transitions()
        sample = self.replay_buffer.sample(1)

        assert sample['observation'].dtype == self.env.observation_space[
            'observation'].dtype
        assert sample['achieved_goal'].dtype == self.env.observation_space[
            'achieved_goal'].dtype
        assert sample['goal'].dtype == self.env.observation_space[
            'desired_goal'].dtype
        assert sample['action'].dtype == self.env.action_space.dtype

    def test_eviction_policy(self):
        self.replay_buffer.add_transitions(observation=[self.obs, self.obs],
                                           action=[1, 2])
        assert not self.replay_buffer.full
        self.replay_buffer.add_transitions(observation=[self.obs, self.obs],
                                           action=[3, 4])
        assert self.replay_buffer.full
        self.replay_buffer.add_transitions(observation=[self.obs, self.obs],
                                           action=[5, 6])
        self.replay_buffer.add_transitions(observation=[self.obs, self.obs],
                                           action=[7, 8])

        assert np.array_equal(self.replay_buffer._buffer['action'],
                              [[7], [8], [6]])
        assert self.replay_buffer.n_transitions_stored == 3

    def test_pickleable(self):
        self._add_transitions()
        replay_buffer_pickled = pickle.loads(pickle.dumps(self.replay_buffer))
        assert replay_buffer_pickled._buffer.keys(
        ) == self.replay_buffer._buffer.keys()
        for k in replay_buffer_pickled._buffer:
            assert replay_buffer_pickled._buffer[
                k].shape == self.replay_buffer._buffer[k].shape
        sample = self.replay_buffer.sample(1)
        sample2 = replay_buffer_pickled.sample(1)
        for k in self.replay_buffer._buffer:
            assert sample[k].shape == sample2[k].shape
