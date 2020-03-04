"""Module for RL2.

This module contains RL2Worker and the environment wrapper for RL2.
"""
import akro
import gym
import numpy as np

from garage.envs.env_spec import EnvSpec
from garage.sampler.worker import DefaultWorker


class RL2Env(gym.Wrapper):
    """Environment wrapper for RL2.

    In RL2, observation is concatenated with previous action,
    reward and terminal signal to form new observation.

    Also, different tasks could have different observation dimension.
    An example is in ML45 from MetaWorld (reference:
    https://arxiv.org/pdf/1910.10897.pdf). This wrapper pads the
    observation to the maximum observation dimension with zeros.

    Args:
        env (gym.Env): An env that will be wrapped.
        max_obs_dim (int): Maximum observation dimension in the environments
             or tasks. Set to None when it is not applicable.

    """

    def __init__(self, env, max_obs_dim=None):
        super().__init__(env)
        self._max_obs_dim = max_obs_dim
        action_space = akro.from_gym(self.env.action_space)
        observation_space = self._create_rl2_obs_space(env)
        self._spec = EnvSpec(action_space=action_space,
                             observation_space=observation_space)

    def _create_rl2_obs_space(self, env):
        """Create observation space for RL2.

        Args:
            env (gym.Env): An env that will be wrapped.

        Returns:
            gym.spaces.Box: Augmented observation space.

        """
        obs_flat_dim = np.prod(env.observation_space.shape)
        action_flat_dim = np.prod(env.action_space.shape)
        if self._max_obs_dim is not None:
            obs_flat_dim = self._max_obs_dim
        return akro.Box(low=-np.inf,
                        high=np.inf,
                        shape=(obs_flat_dim + action_flat_dim + 1 + 1, ))

    # pylint: disable=arguments-differ
    def reset(self):
        """gym.Env reset function.

        Returns:
            np.ndarray: augmented observation.

        """
        obs = self.env.reset()
        # pad zeros if needed for running ML45
        if self._max_obs_dim is not None:
            obs = np.concatenate(
                [obs, np.zeros(self._max_obs_dim - obs.shape[0])])
        return np.concatenate(
            [obs, np.zeros(self.env.action_space.shape), [0], [0]])

    def step(self, action):
        """gym.Env step function.

        Args:
            action (int): action taken.

        Returns:
            np.ndarray: augmented observation.
            float: reward.
            bool: terminal signal.
            dict: environment info.

        """
        next_obs, reward, done, info = self.env.step(action)
        if self._max_obs_dim is not None:
            next_obs = np.concatenate(
                [next_obs,
                 np.zeros(self._max_obs_dim - next_obs.shape[0])])
        next_obs = np.concatenate([next_obs, action, [reward], [done]])
        return next_obs, reward, done, info

    @property
    def spec(self):
        """Environment specification.

        Returns:
            EnvSpec: Environment specification.

        """
        return self._spec


class RL2Worker(DefaultWorker):
    """Initialize a worker for RL2.

    In RL2, policy does not reset between trajectories in each meta batch.
    Policy only resets once at the beginning of a trial/meta batch.

    Args:
        seed(int): The seed to use to intialize random number generators.
        max_path_length(int or float): The maximum length paths which will
            be sampled. Can be (floating point) infinity.
        worker_number(int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
        n_paths_per_trial (int): Number of trajectories sampled per trial/
            meta batch. Policy resets in the beginning of a meta batch,
            and obtain `n_paths_per_trial` trajectories in one meta batch.

    Attributes:
        agent(Policy or None): The worker's agent.
        env(gym.Env or None): The worker's environment.

    """

    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_path_length,
            worker_number,
            n_paths_per_trial=2):
        self._n_paths_per_trial = n_paths_per_trial
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)

    def start_rollout(self):
        """Begin a new rollout."""
        self._path_length = 0
        self._prev_obs = self.env.reset()

    def rollout(self):
        """Sample a single rollout of the agent in the environment.

        Returns:
            garage.TrajectoryBatch: The collected trajectory.

        """
        self.agent.reset()
        for _ in range(self._n_paths_per_trial):
            self.start_rollout()
            while not self.step_rollout():
                pass
        self._agent_infos['batch_idx'] = np.full(len(self._rewards),
                                                 self._worker_number)
        return self.collect_rollout()
