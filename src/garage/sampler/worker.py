"""Worker class used in all Samplers."""
import abc
from collections import defaultdict

import numpy as np

from garage.experiment import deterministic


class Worker(abc.ABC):
    """Worker class used in all Samplers."""

    def __init__(self, *, seed, max_path_length, worker_number):
        """Initialize a worker.

        Args:
            seed(int): The seed to use to intialize random number generators.
            max_path_length(int or float): The maximum length paths which will
                be sampled. Can be (floating point) infinity.
            worker_number(int): The number of the worker this update is
                occurring in. This argument is used to set a different seed for
                each worker.

        Should create fields the following fields:
            agent(Policy or None): The worker's initial agent.
            env(gym.Env or None): The worker's environment.

        """
        self._seed = seed
        self._max_path_length = max_path_length
        self._worker_number = worker_number

    def update_agent(self, agent_update):
        """Update the worker's agent, using agent_update.

        Args:
            agent_update(object): An agent update. The exact type of this
                argument depends on the `Worker` implementation.

        """

    def update_env(self, env_update):
        """Update the worker's env, using env_update.

        Args:
            env_update(object): An environment update. The exact type of this
                argument depends on the `Worker` implementation.

        """

    def rollout(self):
        """Sample a single rollout of the agent in the environment.

        Returns:
            observations(np.array): Non-flattened array of observations. There
                should be one more of these than actions. Note that
                observations[i] (for i < len(observations) - 1) was used by the
                agent to choose actions[i]. Should have shape (T + 1, S^*) (the
                unflattened state space of the current environment).
            actions(np.array): Non-flattened array of actions. Should have
                shape (T, S^*) (the unflattened action space of the current
                environment).
            rewards(np.array): Array of rewards of shape (T,) (1D array of
                length timesteps).
            agent_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `agent_info` arrays.
            env_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `env_info` arrays.

        """

    def start_rollout(self):
        """Begin a new rollout."""

    def step_rollout(self):
        """Take a single time-step in the current rollout.

        Returns:
            True iff the path is done, either due to the environment indicating
            termination of due to reaching `max_path_length`.

        """

    def collect_rollout(self):
        """Collect the current rollout, clearing the internal buffer.

        Returns:
            observations(np.array): Non-flattened array of observations. There
                should be one more of these than actions. Note that
                observations[i] (for i < len(observations) - 1) was used by the
                agent to choose actions[i]. Should have shape (T + 1, S^*) (the
                unflattened state space of the current environment).
            actions(np.array): Non-flattened array of actions. Should have
                shape (T, S^*) (the unflattened action space of the current
                environment).
            rewards(np.array): Array of rewards of shape (T,) (1D array of
                length timesteps).
            agent_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `agent_info` arrays.
            env_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `env_info` arrays.

        """

    def __getstate__(self):
        """Refuse to be pickled.

        Raises:
            ValueError: Always raised, since pickling Workers is not supported.

        """
        raise ValueError('Workers are not pickleable. '
                         'Please pickle the WorkerFactory instead.')


class DefaultWorker(Worker):
    """Initialize a worker.

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
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)
        self.agent = None
        self.env = None
        self._observations = []
        self._actions = []
        self._rewards = []
        self._agent_infos = defaultdict(list)
        self._env_infos = defaultdict(list)
        self._last_obs = None
        self._path_length = 0
        self.worker_init()

    def worker_init(self):
        """Initialize a worker."""
        deterministic.set_seed(self._seed + self._worker_number)

    def update_agent(self, agent_update):
        """Update an agent, assuming it implements garage.Policy.

        Args:
            agent_update(Dict[str, np.array]): Parameters to agent, which
                should have been generated by calling
                `policy.get_param_values`. Note that other implementations of
                `Worker` may take different types for this parameter.

        """
        self.agent.set_param_values(agent_update)

    def update_env(self, env_update):
        """Use any non-None env_update as a new environment.

        A simple env update function. If env_update is not None, it should be
        the complete new environment.

        This allows changing environments by passing the new environment as
        `env_update` into `obtain_samples`.

        Args:
            env_update(gym.Env or None): The environment to replace the
                existing env with. Note that other implementations of
                `Worker` may take different types for this parameter.

        """
        if env_update is not None:
            self.env.close()
            self.env = env_update

    def start_rollout(self):
        """Begin a new rollout."""
        self._observations = []
        self._actions = []
        self._rewards = []
        self._agent_infos = defaultdict(list)
        self._env_infos = defaultdict(list)
        self._path_length = 0
        self._last_obs = self.env.reset()
        self.agent.reset()

    def step_rollout(self):
        """Take a single time-step in the current rollout.

        Returns:
            bool: True iff the path is done, either due to the environment
            indicating termination of due to reaching `max_path_length`.

        """
        if self._path_length < self._max_path_length:
            a, agent_info = self.agent.get_action(self._last_obs)
            next_o, r, d, env_info = self.env.step(a)
            self._observations.append(self._last_obs)
            self._rewards.append(r)
            self._actions.append(a)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            for k, v in env_info.items():
                self._env_infos[k].append(v)
            self._path_length += 1
            if d:
                return True
            self._last_obs = next_o
            return False
        else:
            return True

    def collect_rollout(self):
        """Collect the current rollout, clearing the internal buffer.

        Returns:
            Tuple[np.ndarray or Dict[str, np.ndarray]]
            observations(np.array): Non-flattened array of observations. There
                should be one more of these than actions. Note that
                observations[i] (for i < len(observations) - 1) was used by the
                agent to choose actions[i]. Should have shape (T + 1, S^*) (the
                unflattened state space of the current environment).
            actions(np.array): Non-flattened array of actions. Should have
                shape (T, S^*) (the unflattened action space of the current
                environment).
            rewards(np.array): Array of rewards of shape (T,) (1D array of
                length timesteps).
            agent_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `agent_info` arrays.
            env_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `env_info` arrays.

        """
        observations = self._observations
        self._observations = []
        actions = self._actions
        self._actions = []
        rewards = self._rewards
        self._rewards = []
        agent_infos = self._agent_infos
        self._agent_infos = defaultdict(list)
        env_infos = self._env_infos
        self._env_infos = defaultdict(list)
        for k, v in agent_infos.items():
            agent_infos[k] = np.asarray(v)
        for k, v in env_infos.items():
            env_infos[k] = np.asarray(v)
        return (np.array(observations), np.array(actions), np.array(rewards),
                dict(agent_infos), dict(env_infos))

    def rollout(self):
        """Sample a single rollout of the agent in the environment.

        Returns:
            Tuple[np.ndarray or Dict[str, np.ndarray]]
            observations(np.array): Non-flattened array of observations. There
                should be one more of these than actions. Note that
                observations[i] (for i < len(observations) - 1) was used by the
                agent to choose actions[i]. Should have shape (T + 1, S^*) (the
                unflattened state space of the current environment).
            actions(np.array): Non-flattened array of actions. Should have
                shape (T, S^*) (the unflattened action space of the current
                environment).
            rewards(np.array): Array of rewards of shape (T,) (1D array of
                length timesteps).
            agent_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `agent_info` arrays.
            env_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `env_info` arrays.

        """
        self.start_rollout()
        while not self.step_rollout():
            pass
        return self.collect_rollout()
