"""
This module implements a replay buffer memory.

Replay buffer is an important technique in reinforcement learning. It
stores transitions in a memory buffer of fixed size. When the buffer is
full, oldest memory will be discarded. At each step, a batch of memories
will be sampled from the buffer to update the agent's parameters. In a
word, replay buffer breaks temporal correlations and thus benefits RL
algorithms.
"""

import abc
from abc import abstractmethod

import numpy as np


class ReplayBuffer(metaclass=abc.ABCMeta):
    """Abstract class for Replay Buffer."""

    def __init__(self,
                 env_spec,
                 size_in_transitions,
                 time_horizon,
                 dtype=np.float32):
        """
        Initialize the data used in ReplayBuffer.

        :param buffer_shapes: shape of values for each key in the buffer
        :param size_in_transitions: total size of transitions in the buffer
        :param time_horizon: time horizon of rollout
        """
        self._current_size = 0
        self._n_transitions_stored = 0
        self._time_horizon = time_horizon
        self._size = size_in_transitions // time_horizon
        self._initialized_buffer = False
        self._buffer = {}
        self._episode_buffer = {}
        self._dtype = dtype

    def store_episode(self):
        """Add an episode to the buffer."""
        episode_buffer = self._convert_episode_to_batch_major()
        rollout_batch_size = len(episode_buffer["observation"])
        idx = self._get_storage_idx(rollout_batch_size)

        for key in self._buffer.keys():
            self._buffer[key][idx] = episode_buffer[key]
        self._n_transitions_stored += self._time_horizon * rollout_batch_size

    @abstractmethod
    def sample(self, batch_size):
        """Sample a transition of batch_size."""
        raise NotImplementedError

    def add_transition(self, **kwargs):
        """Add one transition into the replay buffer."""
        if not self._initialized_buffer:
            self._initialize_buffer(**kwargs)

        for key, value in kwargs.items():
            self._episode_buffer[key].append(value)

        if len(self._episode_buffer["observation"]) == self._time_horizon:
            self.store_episode()
            for key in self._episode_buffer.keys():
                self._episode_buffer[key].clear()

    def _initialize_buffer(self, **kwargs):
        for key, value in kwargs.items():
            self._episode_buffer[key] = list()
            self._buffer[key] = np.zeros(
                [self._size, self._time_horizon, *np.array(value).shape[1:]],
                dtype=self._dtype)
        self._initialized_buffer = True

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
        """
        Convert the shape of episode_buffer.

        episode_buffer: {time_horizon, algo.rollout_batch_size, flat_dim}.
        buffer: {size, time_horizon, flat_dim}.
        """
        transitions = {}
        for key in self._episode_buffer.keys():
            val = np.array(self._episode_buffer[key])
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
