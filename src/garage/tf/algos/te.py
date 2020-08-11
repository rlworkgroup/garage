"""Task Embedding Algorithm."""
from collections import defaultdict

import numpy as np

from garage import EpisodeBatch, StepType
from garage.sampler import DefaultWorker


class TaskEmbeddingWorker(DefaultWorker):
    """A sampler worker for Task Embedding Algorithm.

    In addition to DefaultWorker, this worker adds one-hot task id to env_info,
    and adds latent and latent infos to agent_info.

    Args:
        seed (int): The seed to use to intialize random number generators.
        max_episode_length (int or float): The maximum length of episodes to
            sample. Can be (floating point) infinity.
        worker_number(int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.

    Attributes:
        agent (Policy or None): The worker's agent.
        env (Environment or None): The worker's environment.

    """

    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_episode_length,
            worker_number):
        self._latents = []
        self._tasks = []
        self._latent_infos = defaultdict(list)
        self._z, self._t, self._latent_info = None, None, None
        super().__init__(seed=seed,
                         max_episode_length=max_episode_length,
                         worker_number=worker_number)

    def start_episode(self):
        """Begin a new episode."""
        # pylint: disable=protected-access
        self._t = self.env._active_task_one_hot()
        self._z, self._latent_info = self.agent.get_latent(self._t)
        self._z = self.agent.latent_space.flatten(self._z)
        super().start_episode()

    def step_episode(self):
        """Take a single time-step in the current episode.

        Returns:
            bool: True iff the episode is done, either due to the environment
                indicating termination of due to reaching `max_episode_length`.

        """
        if self._eps_length < self._max_episode_length:
            a, agent_info = self.agent.get_action_given_latent(
                self._prev_obs, self._z)
            es = self.env.step(a)
            self._observations.append(self._prev_obs)
            self._env_steps.append(es)
            self._tasks.append(self._t)
            self._latents.append(self._z)
            for k, v in self._latent_info.items():
                self._latent_infos[k].append(v)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            self._eps_length += 1

            if not es.last:
                self._prev_obs = es.observation
                return False

        self._lengths.append(self._eps_length)
        self._last_observations.append(self._prev_obs)

        return True

    def collect_episode(self):
        """Collect the current episode, clearing the internal buffer.

        One-hot task id is saved in env_infos['task_onehot']. Latent is saved
        in agent_infos['latent']. Latent infos are saved in
        agent_infos['latent_info_name'], where info_name is the original latent
        info name.

        Returns:
            EpisodeBatch: A batch of the episodes completed since the last call
                to collect_episode().

        """
        observations = self._observations
        self._observations = []
        last_observations = self._last_observations
        self._last_observations = []

        actions = []
        rewards = []
        env_infos = defaultdict(list)
        step_types = []

        for es in self._env_steps:
            actions.append(es.action)
            rewards.append(es.reward)
            step_types.append(es.step_type)
            for k, v in es.env_info.items():
                env_infos[k].append(v)
        self._env_steps = []

        latents = self._latents
        self._latents = []
        tasks = self._tasks
        self._tasks = []

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

        return EpisodeBatch(env_spec=self.env.spec,
                            observations=np.asarray(observations),
                            last_observations=np.asarray(last_observations),
                            actions=np.asarray(actions),
                            rewards=np.asarray(rewards),
                            step_types=np.asarray(step_types, dtype=StepType),
                            env_infos=(env_infos),
                            agent_infos=dict(agent_infos),
                            lengths=np.asarray(lengths, dtype='i'))
