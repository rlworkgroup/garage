"""A replay buffer memory for meta RL."""

import numpy as np


class MetaReplayBuffer:
    """This class implements MetaReplayBuffer.

    It stores information of each sample in separate numpy arrays.

    Args:
        max_replay_buffer_size (int): Maximum buffer size.
        observation_dim (numpy.ndarray): Dimension of observations.
        action_dim (numpy.ndarray): Dimension of actions.

    """

    def __init__(self, max_replay_buffer_size, observation_dim, action_dim):

        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros(
            (max_replay_buffer_size, observation_dim))
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation):
        """Add a sample to the buffer.

        Args:
            observation (numpy.ndarray): Observation.
            action (numpy.ndarray): Action.
            reward (float): Reward.
            terminal (bool): Terminal state.
            next_observation (numpy.ndarray): Next obseravation.

        """
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._advance()

    def add_path(self, path):
        """Add a path to buffer.

        Args:
            path (dict): Dictionary containing path information.

        """
        for _, (obs, action, reward, next_obs, terminal) in enumerate(
                zip(path['observations'], path['actions'], path['rewards'],
                    path['next_observations'], path['terminals'])):
            self.add_sample(obs, action, reward, terminal, next_obs)
        self.terminate_episode()

    def size(self):
        """Get size of buffer.

        Returns:
            int: Current size of buffer.

        """
        return self._size

    def _advance(self):
        """Increment size of buffer."""
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def terminate_episode(self):
        """Terminate current episode."""
        # store buffer position of the start of current episode
        self._episode_starts.append(self._cur_episode_start)
        self._cur_episode_start = self._top

    def clear(self):
        """Clear buffer."""
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def sample_data(self, indices):
        """Sample data from buffer given indices.

        Args:
            indices (list): List of indices indicating which samples to take
                from buffer.

        Returns:
            dict: Dictionary containing samples.

        """
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )

    def random_batch(self, batch_size):
        """Sample a batch of random unordered transitions from buffer.

        Args:
            batch_size (int): Size of random batch.

        Returns:
            dict: Dictionary containing random batch.

        """
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        """Sample a batch of random ordered transitions from buffer.

        Args:
            batch_size (int): Size of random batch.

        Returns:
            dict: Dictionary containing random batch.

        """
        i = 0
        indices = []
        while len(indices) < batch_size:
            start = np.random.choice(self._episode_starts[:-1])
            pos_idx = self._episode_starts.index(start)
            indices += list(range(start, self._episode_starts[pos_idx + 1]))
            i += 1
        indices = indices[:batch_size]

        return self.sample_data(indices)
