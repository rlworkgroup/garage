"""
This module implements a Hindsight Experience Replay (HER).

See: https://arxiv.org/abs/1707.01495.
"""
import inspect

import numpy as np


def make_her_sample(replay_k, reward_fun):
    """
    Generate a transition sampler for HER ReplayBuffer.

    :param replay_k: the ratio between HER replays and regular replays
    :param reward_fun: function to re-compute the reward with substituted goals
    :return:
    """
    future_p = 1 - (1. / (1 + replay_k))

    def _her_sample_transitions(episode_batch, sample_batch_size):
        """
        Generate a dictionary of transitions.

        :param episode_batch: [batch_size, T, dim]
        :param sample_batch_size: batch_size per sample.
        :return: transitions which transitions[key] has the shape of
        [sample_batch_size, dim].
        """
        # Select which episodes to use
        time_horizon = episode_batch["action"].shape[1]
        rollout_batch_size = episode_batch["action"].shape[0]
        episode_idxs = np.random.randint(0, rollout_batch_size,
                                         sample_batch_size)
        # Select time steps to use
        t_samples = np.random.randint(time_horizon, size=sample_batch_size)
        transitions = {
            key: episode_batch[key][episode_idxs, t_samples].copy()
            for key in episode_batch.keys()
        }

        her_indexes = np.where(
            np.random.uniform(size=sample_batch_size) < future_p)
        future_offset = np.random.uniform(size=sample_batch_size) * (
            time_horizon - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        future_ag = episode_batch["achieved_goal"][episode_idxs[her_indexes],
                                                   future_t]
        transitions["goal"][her_indexes] = future_ag

        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith("info_"):
                info[key.replace("info_", "")] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params_keys = inspect.signature(reward_fun).parameters.keys()
        reward_params = {
            rk: transitions[k]
            for k, rk in zip(["next_achieved_goal", "goal"],
                             list(reward_params_keys)[:-1])
        }
        reward_params["info"] = info
        transitions["reward"] = reward_fun(**reward_params)

        transitions = {
            k: transitions[k].reshape(sample_batch_size,
                                      *transitions[k].shape[1:])
            for k in transitions.keys()
        }
        assert (transitions["action"].shape[0] == sample_batch_size)
        return transitions

    return _her_sample_transitions


class HerReplayBuffer:
    """This class implements HerReplayBuffer."""

    def __init__(self, buffer_shapes, size_in_transitions, time_horizon,
                 sample_transitions):
        """
        Initialize the data used in HER.

        :param buffer_shapes: shape of values for each key in the buffer
        :param size_in_transitions: total size of transitions in the buffer
        :param time_horizon: time horizon of rollout
        :param sample_transitions: function to sample transitions
        """
        self._current_size = 0
        self._n_transitions_stored = 0
        self._time_horizon = time_horizon
        self._sample_transitions = sample_transitions
        self._episode_buffer = {}
        self._size = size_in_transitions // time_horizon
        for key in buffer_shapes.keys():
            self._episode_buffer[key] = list()
        self._buffer = {
            key: np.empty([self._size, *shape])
            for key, shape in buffer_shapes.items()
        }

    def sample(self, batch_size):
        """Sample a transition of batch_size."""
        buffer = {}
        for key in self._buffer.keys():
            buffer[key] = self._buffer[key][:self._current_size]
        buffer["next_observation"] = buffer["observation"][:, 1:, :]
        buffer["next_achieved_goal"] = buffer["achieved_goal"][:, 1:, :]

        transitions = self._sample_transitions(buffer, batch_size)

        for key in (["reward", "next_observation", "next_achieved_goal"] +
                    list(self._buffer.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self):
        """Add an episode to the buffer."""
        idx = self._get_storage_idx()
        for key in self._buffer.keys():
            self._buffer[key][idx] = np.array(self._episode_buffer[key])
        self._n_transitions_stored += self._time_horizon

    def add_transition(self, **kwargs):
        """Add one transition into the replay buffer."""
        for key, value in kwargs.items():
            self._episode_buffer[key].append(kwargs[key])

        if len(self._episode_buffer["observation"]) == self._time_horizon + 1:
            self.store_episode()
            for key in self._episode_buffer.keys():
                self._episode_buffer[key].clear()

    @property
    def full(self):
        """Whether the buffer is full."""
        return self._current_size == self._size

    def _get_storage_idx(self):
        """Get the storage index for the episode to add into the buffer."""
        size_increment = 1
        if self._current_size + size_increment <= self._size:
            idx = np.arange(self._current_size,
                            self._current_size + size_increment)
        elif self._current_size < self._size:
            overflow = size_increment - (self._size - self._current_size)
            idx_a = np.arange(self._current_size, self._size)
            idx_b = np.random.randint(0, self._current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self._size, size_increment)

        # Update replay size
        self._current_size = min(self._size,
                                 self._current_size + size_increment)

        if size_increment == 1:
            idx = idx[0]
        return idx
