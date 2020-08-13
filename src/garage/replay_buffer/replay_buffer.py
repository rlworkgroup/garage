"""This module implements a replay buffer memory.

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
    """Abstract class for Replay Buffer.

    Args:
        env_spec (EnvSpec): Environment specification.
        size_in_transitions (int): total size of transitions in the buffer
        time_horizon (int): time horizon of epsiode.

    """

    def __init__(self, env_spec, size_in_transitions, time_horizon):
        del env_spec
        self._current_size = 0
        self._current_ptr = 0
        self._n_transitions_stored = 0
        self._time_horizon = time_horizon
        self._size_in_transitions = size_in_transitions
        self._size = size_in_transitions // time_horizon
        self._initialized_buffer = False
        self._buffer = {}
        self._episode_buffer = {}

    def store_episode(self):
        """Add an episode to the buffer."""
        episode_buffer = self._convert_episode_to_batch_major()
        episode_batch_size = len(episode_buffer['observation'])
        idx = self._get_storage_idx(episode_batch_size)

        for key in self._buffer:
            self._buffer[key][idx] = episode_buffer[key]
        self._n_transitions_stored = min(
            self._size_in_transitions, self._n_transitions_stored +
            self._time_horizon * episode_batch_size)

    @abstractmethod
    def sample(self, batch_size):
        """Sample a transition of batch_size.

        Args:
            batch_size(int): The number of transitions to be sampled.

        """
        raise NotImplementedError

    def add_transition(self, **kwargs):
        """Add one transition into the replay buffer.

        Args:
            kwargs (dict(str, [numpy.ndarray])): Dictionary that holds
                the transitions.

        """
        transition = {k: [v] for k, v in kwargs.items()}
        self.add_transitions(**transition)

    def add_transitions(self, **kwargs):
        """Add multiple transitions into the replay buffer.

        A transition contains one or multiple entries, e.g.
        observation, action, reward, terminal and next_observation.
        The same entry of all the transitions are stacked, e.g.
        {'observation': [obs1, obs2, obs3]} where obs1 is one
        numpy.ndarray observation from the environment.

        Args:
            kwargs (dict(str, [numpy.ndarray])): Dictionary that holds
                the transitions.

        """
        if not self._initialized_buffer:
            self._initialize_buffer(**kwargs)

        for key, value in kwargs.items():
            self._episode_buffer[key].append(value)

        if len(self._episode_buffer['observation']) == self._time_horizon:
            self.store_episode()
            for key in self._episode_buffer:
                self._episode_buffer[key].clear()

    def _initialize_buffer(self, **kwargs):
        for key, value in kwargs.items():
            self._episode_buffer[key] = list()
            values = np.array(value)
            self._buffer[key] = np.zeros(
                [self._size, self._time_horizon, *values.shape[1:]],
                dtype=values.dtype)
        self._initialized_buffer = True

    def _get_storage_idx(self, size_increment=1):
        """Get the storage index for the episode to add into the buffer.

        Args:
            size_increment(int): The number of storage indeces that new
                transitions will be placed in.

        Returns:
            numpy.ndarray: The indeces to store size_incremente transitions at.

        """
        if self._current_size + size_increment <= self._size:
            idx = np.arange(self._current_size,
                            self._current_size + size_increment)
        elif self._current_size < self._size:
            overflow = size_increment - (self._size - self._current_size)
            idx_a = np.arange(self._current_size, self._size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self._current_ptr = overflow
        else:
            if self._current_ptr + size_increment <= self._size:
                idx = np.arange(self._current_ptr,
                                self._current_ptr + size_increment)
                self._current_ptr += size_increment
            else:
                overflow = size_increment - (self._size - self._current_size)
                idx_a = np.arange(self._current_ptr, self._size)
                idx_b = np.arange(0, overflow)
                idx = np.concatenate([idx_a, idx_b])
                self._current_ptr = overflow

        # Update replay size
        self._current_size = min(self._size,
                                 self._current_size + size_increment)

        if size_increment == 1:
            idx = idx[0]

        return idx

    def _convert_episode_to_batch_major(self):
        """Convert the shape of episode_buffer.

        episode_buffer: {time_horizon, algo.episode_batch_size, flat_dim}.
        buffer: {size, time_horizon, flat_dim}.

        Returns:
            dict: Transitions that have been formated to fit properly in this
                replay buffer.

        """
        transitions = {}
        for key in self._episode_buffer:
            val = np.array(self._episode_buffer[key])
            transitions[key] = val.swapaxes(0, 1)
        return transitions

    @property
    def full(self):
        """Whether the buffer is full.

        Returns:
            bool: True of the buffer has reachd its maximum size.
                False otherwise.

        """
        return self._current_size == self._size

    @property
    def n_transitions_stored(self):
        """Return the size of the replay buffer.

        Returns:
            int: Size of the current replay buffer.

        """
        return self._n_transitions_stored
