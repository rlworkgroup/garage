"""Worker interface used in all Samplers."""
import abc


class Worker(abc.ABC):
    """Worker class used in all Samplers."""

    def __init__(self, *, seed, max_episode_length, worker_number):
        """Initialize a worker.

        Args:
            seed (int): The seed to use to intialize random number generators.
            max_episode_length (int or float): The maximum length of episodes
                which will be sampled. Can be (floating point) infinity.
            worker_number (int): The number of the worker this update is
                occurring in. This argument is used to set a different seed for
                each worker.

        Should create fields the following fields:
            agent (Policy or None): The worker's initial agent.
            env (Environment or None): The worker's environment.

        """
        self._seed = seed
        self._max_episode_length = max_episode_length
        self._worker_number = worker_number

    def update_agent(self, agent_update):
        """Update the worker's agent, using agent_update.

        Args:
            agent_update (object): An agent update. The exact type of this
                argument depends on the `Worker` implementation.

        """

    def update_env(self, env_update):
        """Update the worker's env, using env_update.

        Args:
            env_update (object): An environment update. The exact type of this
                argument depends on the `Worker` implementation.

        """

    def rollout(self):
        """Sample a single episode of the agent in the environment.

        Returns:
            EpisodeBatch: Batch of sampled episodes. May be truncated if
                max_episode_length is set.

        """

    def start_episode(self):
        """Begin a new episode."""

    def step_episode(self):
        """Take a single time-step in the current episode.

        Returns:
            True iff the episode is done, either due to the environment
            indicating termination of due to reaching `max_episode_length`.

        """

    def collect_episode(self):
        """Collect the current episode, clearing the internal buffer.

        Returns:
            EpisodeBatch: Batch of sampled episodes. May be truncated if the
               episodes haven't completed yet.

        """

    def shutdown(self):
        """Shutdown the worker."""

    def __getstate__(self):
        """Refuse to be pickled.

        Raises:
            ValueError: Always raised, since pickling Workers is not supported.

        """
        raise ValueError('Workers are not pickleable. '
                         'Please pickle the WorkerFactory instead.')
