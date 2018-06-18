import numpy as np


class ReplayBuffer(object):
    """
    This class caches transitions in the training of RL algorithms.
    It uses random batch sample to minimize correlations between samples.
    """

    def __init__(self, max_buffer_size, observation_dim, action_dim):
        """
        The init method initializes the data in a transition, which includes
        observation, action, reward, terminal and next_observation.

        Args:
            max_buffer_size(int): Max size of the buffer cache.
            observation_dim(int): Observation space dimension.
            action_dim(int): Action space dimension.
        """
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_buffer_size = max_buffer_size
        self._observations = np.zeros((max_buffer_size, observation_dim))
        self._next_observations = np.zeros((max_buffer_size, observation_dim))
        self._actions = np.zeros((max_buffer_size, action_dim))
        self._rewards = np.zeros(max_buffer_size)
        self._terminals = np.zeros(max_buffer_size)

        self._top = 0
        self._size = 0

    def random_sample(self, sample_size):
        """
        Randomly samples a batch of transitions.

        Args:
            sample_size(int): Size of the sampled batch.

        Returns:
            A dictionary contains fields and values in the sampled transitions.
        """
        assert self._size > sample_size
        indices = np.random.choice(self._size, size=sample_size)
        return {
            "observations": self._observations[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "terminals": self._terminals[indices],
            "next_observations": self._next_observations[indices],
        }

    def add_transition(self, observation, action, reward, terminal,
                       next_observation):
        """
        Add one transition into the replay buffer.

        Args:
            observation:
            action:
            reward:
            terminal:
            next_observation:

        Returns:
        """
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_observations[self._top] = next_observation

        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    @property
    def size(self):
        return self._size
