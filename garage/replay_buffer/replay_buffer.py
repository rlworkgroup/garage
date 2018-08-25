"""
This module implements a replay buffer memory.

Replay buffer is an important technique in reinforcement learning. It
stores transitions in a memory buffer of fixed size. When the buffer is
full, oldest memory will be discarded. At each step, a batch of memories
will be sampled from the buffer to update the agent's parameters. In a
word, replay buffer breaks temporal correlations and thus benefits RL
algorithms.
"""
import numpy as np


class ReplayBuffer:
    """
    This class caches transitions in the training of RL algorithms.

    It uses random batch sample to minimize correlations between samples.
    """

    def __init__(self, buffer_shapes, max_buffer_size):
        """
        Initialize the data in a transition.

        Args:
            max_buffer_size(int): Max size of the buffer cache.
        """
        self._max_buffer_size = max_buffer_size

        self._buffer = {
            key: np.empty((max_buffer_size, *shape))
            for key, shape in buffer_shapes.items()
        }

        self._top = 0
        self._size = 0

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions.

        Args:
            batch_size(int): Size of the sampled batch.

        Returns:
            A dictionary contains fields and values in the sampled transitions.

        """
        assert self._size > batch_size
        indices = np.random.choice(self._size, size=batch_size)

        return {key: self._buffer[key][indices] for key in self._buffer.keys()}

    def add_transition(self, **kwargs):
        for key, value in kwargs.items():
            self._buffer[key][self._top] = kwargs[key]

        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    @property
    def size(self):
        """
        Return the size of the replay buffer.

        Returns:
            self._size: Size of the current replay buffer.

        """
        return self._size
