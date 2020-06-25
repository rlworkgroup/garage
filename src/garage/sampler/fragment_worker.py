"""Worker that "vectorizes" environments."""
import copy

import numpy as np

from garage import TrajectoryBatch
from garage.sampler import _apply_env_update, InProgressTrajectory
from garage.sampler.default_worker import DefaultWorker


class FragmentWorker(DefaultWorker):
    """Vectorized Worker that collects partial trajectories.

    Useful for off-policy RL.

    Args:
        seed(int): The seed to use to intialize random number generators.
        max_path_length(int or float): The maximum length paths which
            will be sampled. Can be (floating point) infinity.
        length of fragments before they're transmitted out of
        worker_number(int): The number of the worker this update is
            occurring in. This argument is used to set a different seed for
            each worker.
        n_envs (int): Number of environment copies to use.
        timesteps_per_call (int): Maximum number of timesteps to gather per env
            per call to the worker. Defaults to 1 (i.e. gather 1 timestep per
            env each call, or n_envs timesteps in total each call).

    """

    DEFAULT_N_ENVS = 8

    def __init__(self,
                 *,
                 seed,
                 max_path_length,
                 worker_number,
                 n_envs=DEFAULT_N_ENVS,
                 timesteps_per_call=1):
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)
        self._n_envs = n_envs
        self._timesteps_per_call = timesteps_per_call
        self._needs_env_reset = True
        self._envs = [None] * n_envs
        self._agents = [None] * n_envs
        self._path_lengths = [0] * self._n_envs
        self._complete_fragments = []
        # Initialized in start_rollout
        self._fragments = None

    def update_env(self, env_update):
        """Update the environments.

        If passed a list (*inside* this list passed to the Sampler itself),
        distributes the environments across the "vectorization" dimension.

        Args:
            env_update(gym.Env or EnvUpdate or None): The environment to
                replace the existing env with. Note that other implementations
                of `Worker` may take different types for this parameter.

        Raises:
            TypeError: If env_update is not one of the documented types.
            ValueError: If the wrong number of updates is passed.

        """
        if isinstance(env_update, list):
            if len(env_update) != self._n_envs:
                raise ValueError('If separate environments are passed for '
                                 'each worker, there must be exactly n_envs '
                                 '({}) environments, but received {} '
                                 'environments.'.format(
                                     self._n_envs, len(env_update)))
        elif env_update is not None:
            env_update = [
                copy.deepcopy(env_update) for _ in range(self._n_envs)
            ]
        if env_update:
            for env_index, env_up in enumerate(env_update):
                self._envs[env_index], up = _apply_env_update(
                    self._envs[env_index], env_up)
                self._needs_env_reset |= up

    def start_rollout(self):
        """Resets all agents if the environment was updated."""
        if self._needs_env_reset:
            self._needs_env_reset = False
            self.agent.reset([True] * len(self._envs))
            self._path_lengths = [0] * len(self._envs)
            self._fragments = [InProgressTrajectory(env) for env in self._envs]

    def step_rollout(self):
        """Take a single time-step in the current rollout.

        Returns:
            bool: True iff at least one of the paths was completed.

        """
        prev_obs = np.asarray([frag.last_obs for frag in self._fragments])
        actions, agent_infos = self.agent.get_actions(prev_obs)
        completes = [False] * len(self._envs)
        for i, action in enumerate(actions):
            frag = self._fragments[i]
            if self._path_lengths[i] < self._max_path_length:
                agent_info = {k: v[i] for (k, v) in agent_infos.items()}
                frag.step(action, agent_info)
                self._path_lengths[i] += 1
            if (self._path_lengths[i] >= self._max_path_length
                    or frag.terminals[-1]):
                self._path_lengths[i] = 0
                complete_frag = frag.to_batch()
                self._complete_fragments.append(complete_frag)
                self._fragments[i] = InProgressTrajectory(self._envs[i])
                completes[i] = True
        if any(completes):
            self.agent.reset(completes)
        return any(completes)

    def collect_rollout(self):
        """Gather fragments from all in-progress rollouts.

        Returns:
            garage.TrajectoryBatch: A batch of the trajectory fragments.

        """
        for i, frag in enumerate(self._fragments):
            assert frag.env is self._envs[i]
            if len(frag.rewards) > 0:
                complete_frag = frag.to_batch()
                self._complete_fragments.append(complete_frag)
                self._fragments[i] = InProgressTrajectory(
                    frag.env, frag.last_obs)
        assert len(self._complete_fragments) > 0
        result = TrajectoryBatch.concatenate(*self._complete_fragments)
        self._complete_fragments = []
        return result

    def rollout(self):
        """Sample a single rollout of the agent in the environment.

        Returns:
            garage.TrajectoryBatch: The collected trajectory.

        """
        self.start_rollout()
        for _ in range(self._timesteps_per_call):
            self.step_rollout()
        complete_frag = self.collect_rollout()
        return complete_frag

    def shutdown(self):
        """Close the worker's environments."""
        for env in self._envs:
            env.close()
