"""This module implements a simple replay buffer."""
import numpy as np

from garage.replay_buffer.base import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):
    """
    This class implements SimpleReplayBuffer.

    It uses random batch sample to minimize correlations between samples.
    """

    def sample(self, batch_size):
        """Sample a transition of batch_size."""
        assert self._n_transitions_stored >= batch_size
        buffer = {}
        for key in self._buffer.keys():
            buffer[key] = self._buffer[key][:self._current_size]

        # Select which episodes to use
        time_horizon = buffer['action'].shape[1]
        rollout_batch_size = buffer['action'].shape[0]
        episode_idxs = np.random.randint(rollout_batch_size, size=batch_size)
        # Select time steps to use
        t_samples = np.random.randint(time_horizon, size=batch_size)

        transitions = {}
        for key in buffer.keys():
            samples = buffer[key][episode_idxs, t_samples]
            transitions[key] = samples.reshape(batch_size, *samples.shape[1:])

        assert (transitions['action'].shape[0] == batch_size)
        return transitions
