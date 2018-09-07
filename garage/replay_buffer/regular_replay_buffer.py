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

from garage.misc.overrides import overrides
from garage.replay_buffer.base import ReplayBuffer


class RegularReplayBuffer(ReplayBuffer):
    """
    This class caches transitions in the training of RL algorithms.

    It uses random batch sample to minimize correlations between samples.
    """

    def __init__(self, **kwargs):
        super(RegularReplayBuffer, self).__init__(**kwargs)

    @overrides
    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions.

        Args:
            batch_size(int): Size of the sampled batch.

        Returns:
            A dictionary contains fields and values in the sampled transitions.

        """
        assert self._n_transitions_stored > batch_size
        buffer = {}
        for key in self._buffer.keys():
            buffer[key] = self._buffer[key][:self._current_size]

        # Select which episodes to use
        time_horizon = buffer["action"].shape[1]
        rollout_batch_size = buffer["action"].shape[0]
        episode_idxs = np.random.randint(rollout_batch_size, size=batch_size)
        # Select time steps to use
        t_samples = np.random.randint(time_horizon, size=batch_size)

        transitions = {}
        for key in buffer.keys():
            samples = buffer[key][episode_idxs, t_samples].copy()
            transitions[key] = samples.reshape(batch_size, *samples.shape[1:])

        assert (transitions["action"].shape[0] == batch_size)
        return transitions

    @overrides
    def add_transition(self, **kwargs):
        """Add one transition into the replay buffer."""
        for key, value in kwargs.items():
            self._episode_buffer[key].append(kwargs[key])

        if len(self._episode_buffer["observation"]) == self._time_horizon:
            self.store_episode()
            for key in self._episode_buffer.keys():
                self._episode_buffer[key].clear()
