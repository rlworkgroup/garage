from enum import Enum, unique

import numpy as np


@unique
class Buffer(Enum):
    REGULAR = "regular"
    HER = "her"


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, time_horizon):
        """
        Initialize the data used in HER.

        :param buffer_shapes: shape of values for each key in the buffer
        :param size_in_transitions: total size of transitions in the buffer
        :param time_horizon: time horizon of rollout
        """
        self._current_size = 0
        self._n_transitions_stored = 0
        self._time_horizon = time_horizon
        self._episode_buffer = {}
        self._size = size_in_transitions // time_horizon
        for key in buffer_shapes.keys():
            self._episode_buffer[key] = list()
        self._buffer = {
            key: np.zeros([self._size, *shape])
            for key, shape in buffer_shapes.items()
        }

    def store_episode(self):
        """Add an episode to the buffer."""
        episode_buffer = self._convert_episode_to_batch_major()
        rollout_batch_size = len(episode_buffer["observation"])
        idx = self._get_storage_idx(rollout_batch_size)
        for key in self._buffer.keys():
            self._buffer[key][idx] = episode_buffer[key]
        self._n_transitions_stored += self._time_horizon * rollout_batch_size

    def sample(self, batch_size):
        """Sample a transition of batch_size."""
        raise NotImplementedError

    def add_transition(self, **kwargs):
        """Add one transition into the replay buffer."""
        raise NotImplementedError

    def _get_storage_idx(self, size_increment=1):
        """Get the storage index for the episode to add into the buffer."""
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

    def _convert_episode_to_batch_major(self):
        transitions = {}
        for key in self._episode_buffer.keys():
            val = np.array(self._episode_buffer[key]).copy()
            transitions[key] = val.swapaxes(0, 1)
        return transitions

    @property
    def full(self):
        """Whether the buffer is full."""
        return self._current_size == self._size

    @property
    def n_transitions_stored(self):
        """
        Return the size of the replay buffer.

        Returns:
            self._size: Size of the current replay buffer.

        """
        return self._n_transitions_stored
