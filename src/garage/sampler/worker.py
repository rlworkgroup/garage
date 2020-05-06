"""Worker interface used in all Samplers."""
import abc


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
            garage.TrajectoryBatch: Batch of sampled trajectories. May be
                truncated if max_path_length is set.

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
            garage.TrajectoryBatch: Batch of sampled trajectories. May be
                truncated if the rollouts haven't completed yet.

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
