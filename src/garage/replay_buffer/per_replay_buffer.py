"""Prioritized Experience Replay."""

import numpy as np

from garage import StepType, TimeStepBatch
from garage.replay_buffer.path_buffer import PathBuffer


class PERReplayBuffer(PathBuffer):
    """Replay buffer for PER (Prioritized Experience Replay).

    PER assigns priorities to transitions in the buffer. Typically
    these priority of each transition is proportional to the corresponding
    loss computed at each update step. The priorities are then used to create
    a probability distribution when sampling such that higher priority
    transitions are sampled more frequently. For more see
    https://arxiv.org/abs/1511.05952.

    Args:
        capacity_in_transitions (int): total size of transitions in the buffer.
        env_spec (EnvSpec): Environment specification.
        total_timesteps (int): Total timesteps the experiment will run for.
            This is used to calculate the beta parameter when sampling.
        alpha (float): hyperparameter that controls the degree of
            prioritization. Typically between [0, 1], where 0 corresponds to
            no prioritization (uniform sampling).
        beta_init (float): Initial value of beta exponent in importance
            sampling. Beta is linearly annealed from beta_init to 1
            over total_timesteps.
    """

    def __init__(self,
                 capacity_in_transitions,
                 total_timesteps,
                 env_spec,
                 alpha=0.6,
                 beta_init=0.5):
        self._alpha = alpha
        self._beta_init = beta_init
        self._total_timesteps = total_timesteps
        self._curr_timestep = 0
        self._priorities = np.zeros((capacity_in_transitions, ), np.float32)
        self._rng = np.random.default_rng()
        super().__init__(capacity_in_transitions, env_spec)

    def sample_timesteps(self, batch_size):
        """Sample a batch of timesteps from the buffer.

        Args:
            batch_size (int): Number of timesteps to sample.

        Returns:
            TimeStepBatch: The batch of timesteps.
            np.ndarray: Weights of the timesteps.
            np.ndarray: Indices of sampled timesteps
                in the replay buffer.

        """
        samples, w, idx = self.sample_transitions(batch_size)
        step_types = np.array([
            StepType.TERMINAL if terminal else StepType.MID
            for terminal in samples['terminals'].reshape(-1)
        ],
                              dtype=StepType)
        return TimeStepBatch(env_spec=self._env_spec,
                             observations=samples['observations'],
                             actions=samples['actions'],
                             rewards=samples['rewards'],
                             next_observations=samples['next_observations'],
                             step_types=step_types,
                             env_infos={},
                             agent_infos={}), w, idx

    def sample_transitions(self, batch_size):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            dict: A dict of arrays of shape (batch_size, flat_dim).
            np.ndarray: Weights of the timesteps.
            np.ndarray: Indices of sampled timesteps
                in the replay buffer.

        """
        priorities = self._priorities
        if self._transitions_stored < self._capacity:
            priorities = self._priorities[:self._transitions_stored]
        probs = priorities**self._alpha
        probs /= probs.sum()
        idx = self._rng.choice(self._transitions_stored,
                               size=batch_size,
                               p=probs)

        beta = self._beta_init + self._curr_timestep * (
            1.0 - self._beta_init) / self._total_timesteps
        beta = min(1.0, beta)
        transitions = {
            key: buf_arr[idx]
            for key, buf_arr in self._buffer.items()
        }

        w = (self._transitions_stored * probs[idx])**(-beta)
        w /= w.max()
        w = np.array(w)

        return transitions, w, idx

    def update_priorities(self, indices, priorities):
        """Update priorities of timesteps.

        Args:
            indices (np.ndarray): Array of indices corresponding to the
                timesteps/priorities to update.
            priorities (list[float]): new priorities to set.

        """
        for idx, priority in zip(indices, priorities):
            self._priorities[int(idx)] = priority

    def add_path(self, path):
        """Add a path to the buffer.

        This differs from the underlying buffer's add_path method
        in that the priorities for the new samples are set to
        the maximum of all priorities in the buffer.

        Args:
            path (dict): A dict of array of shape (path_len, flat_dim).

        """
        path_len = len(path['observations'])
        self._curr_timestep += path_len

        # find the indices where the path will be stored
        first_seg, second_seg = self._next_path_segments(path_len)

        # set priorities for new timesteps = max(self._priorities)
        # or 1 if buffer is empty
        max_priority = self._priorities.max() or 1.
        self._priorities[first_seg.start:first_seg.stop] = max_priority
        if second_seg != range(0, 0):
            self._priorities[second_seg.start:second_seg.stop] = max_priority
        super().add_path(path)
