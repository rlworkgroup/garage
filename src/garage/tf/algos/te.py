"""Task Embedding Algorithm."""
from collections import defaultdict

import numpy as np

from garage import TrajectoryBatch
from garage.sampler import DefaultWorker


class TaskEmbeddingWorker(DefaultWorker):
    """A sampler worker for Task Embedding Algorithm.

    In addition to DefaultWorker, this worker adds one-hot task id to env_info,
    and adds latent and latent infos to agent_info.

    Args:
        seed(int): The seed to use to intialize random number generators.
        max_path_length(int or float): The maximum length paths which will
            be sampled. Can be (floating point) infinity.
        worker_number(int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.

    Attributes:
        agent(Policy or None): The worker's agent.
        env(gym.Env or None): The worker's environment.

    """

    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_path_length,
            worker_number):
        self._latents = []
        self._tasks = []
        self._latent_infos = defaultdict(list)
        self._z, self._t, self._latent_info = None, None, None
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)

    def start_rollout(self):
        """Begin a new rollout."""
        # pylint: disable=protected-access
        self._t = self.env._active_task_one_hot()
        self._z, self._latent_info = self.agent.get_latent(self._t)
        self._z = self.agent.latent_space.flatten(self._z)
        super().start_rollout()

    def step_rollout(self):
        """Take a single time-step in the current rollout.

        Returns:
            bool: True iff the path is done, either due to the environment
                indicating termination of due to reaching `max_path_length`.

        """
        if self._path_length < self._max_path_length:
            a, agent_info = self.agent.get_action_given_latent(
                self._prev_obs, self._z)
            next_o, r, d, env_info = self.env.step(a)
            self._observations.append(self._prev_obs)
            self._rewards.append(r)
            self._actions.append(a)
            self._tasks.append(self._t)
            self._latents.append(self._z)
            for k, v in self._latent_info.items():
                self._latent_infos[k].append(v)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            for k, v in env_info.items():
                self._env_infos[k].append(v)
            self._path_length += 1
            self._terminals.append(d)

            if not d:
                self._prev_obs = next_o
                return False

        self._lengths.append(self._path_length)
        self._last_observations.append(self._prev_obs)

        return True

    def collect_rollout(self):
        """Collect the current rollout, clearing the internal buffer.

        One-hot task id is saved in env_infos['task_onehot']. Latent is saved
        in agent_infos['latent']. Latent infos are saved in
        agent_infos['latent_info_name'], where info_name is the original latent
        info name.

        Returns:
            garage.TrajectoryBatch: A batch of the trajectories completed since
                the last call to collect_rollout().

        """
        observations = self._observations
        self._observations = []
        last_observations = self._last_observations
        self._last_observations = []
        actions = self._actions
        self._actions = []
        rewards = self._rewards
        self._rewards = []
        terminals = self._terminals
        self._terminals = []
        latents = self._latents
        self._latents = []
        tasks = self._tasks
        self._tasks = []
        env_infos = self._env_infos
        self._env_infos = defaultdict(list)
        agent_infos = self._agent_infos
        self._agent_infos = defaultdict(list)
        latent_infos = self._latent_infos
        self._latent_infos = defaultdict(list)
        for k, v in latent_infos.items():
            latent_infos[k] = np.asarray(v)
        for k, v in agent_infos.items():
            agent_infos[k] = np.asarray(v)
        for k, v in env_infos.items():
            env_infos[k] = np.asarray(v)
        env_infos['task_onehot'] = np.asarray(tasks)
        agent_infos['latent'] = np.asarray(latents)
        for k, v in latent_infos.items():
            agent_infos['latent_{}'.format(k)] = v
        lengths = self._lengths
        self._lengths = []

        return TrajectoryBatch(self.env.spec, np.asarray(observations),
                               np.asarray(last_observations),
                               np.asarray(actions), np.asarray(rewards),
                               np.asarray(terminals), dict(env_infos),
                               dict(agent_infos), np.asarray(lengths,
                                                             dtype='i'))
