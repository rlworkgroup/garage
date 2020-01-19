# pylint: disable=consider-using-dict-comprehension
"""A replay buffer memory for meta-RL."""

from gym.spaces import Box, Discrete, Tuple
import numpy as np

from garage.replay_buffer.meta_replay_buffer import MetaReplayBuffer


class MultiTaskReplayBuffer:
    """This buffer is used in meta-RL algorithms involving multiple tasks.

    It contains a list of MetaReplayBuffers that can be accessed by index.

    Args:
        max_replay_buffer_size (int): Maximum buffer size.
        env (garage.envs.GarageEnv): Meta environment.
        tasks (list): A list of task indices.

    """

    def __init__(self, max_replay_buffer_size, env, tasks):

        self.env = env
        self._obs_space = env.observation_space
        self._action_space = env.action_space
        self.task_buffers = {
            i: MetaReplayBuffer(max_replay_buffer_size=max_replay_buffer_size,
                                observation_dim=get_dim(self._obs_space),
                                action_dim=get_dim(self._action_space))
            for i in tasks
        }

    def add_sample(self, task, observation, action, reward, terminal,
                   next_observation):
        """Add a sample to the buffer.

        Args:
            task (int): Task index.
            observation (numpy.ndarrayy): Observation.
            action (numpy.ndarray): Action.
            reward (float): Reward.
            terminal (bool): Terminal state.
            next_observation (numpy.ndarray): Next obseravation.

        """
        if isinstance(self._action_space, Discrete):
            action = np.eye(self._action_space.n)[action]
        self.task_buffers[task].add_sample(observation, action, reward,
                                           terminal, next_observation)

    def add_path(self, task, path):
        """Add path to a task buffer by its index.

        Args:
            task (int): Task index.
            path (dict): Dictionary containing path information.

        """
        self.task_buffers[task].add_path(path)

    def add_paths(self, task, paths):
        """Add paths to a task buffer by its index.

        Args:
            task (int): Task index.
            paths (dict): Dictionary containing multiple paths.

        """
        for path in paths:
            self.task_buffers[task].add_path(path)

    def terminate_episode(self, task):
        """Terminate current episode of task buffer by its index.

        Args:
            task (int): Task index.

        """
        self.task_buffers[task].terminate_episode()

    def clear_buffer(self, task):
        """Clear a task buffer by its index.

        Args:
            task (int): Task index.

        """
        self.task_buffers[task].clear()

    def random_batch(self, task, batch_size, sequence=False):
        """Sample a batch of random unordered transitions from buffer.

        Args:
            task (int): Task index.
            batch_size (int): Size of random batch.
            sequence (bool): True if sampling trajectories.

        Returns:
            dict: Dictionary containing random batch.

        """
        if sequence:
            batch = self.task_buffers[task].sample_trajectory(batch_size)
        else:
            batch = self.task_buffers[task].sample_batch(batch_size)
        return batch

    def num_steps_can_sample(self, task):
        """Get number of steps that can be sampled.

        Args:
            task (int): Task index.

        Returns:
            int: Number of steps that can be sampled.

        """
        return self.task_buffers[task].size()


def get_dim(space):
    """Get dimension of a space.

    Args:
        space (gym.spaces): Gym space.

    Returns:
        int: Number of steps that can be sampled.

    Raises:
        TypeError: If space is unknown.

    """
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError('Unknown space: {}'.format(space))
