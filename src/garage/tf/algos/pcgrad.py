"""PCGrad worker."""
from garage.sampler import DefaultWorker


class PCGradWorker(DefaultWorker):
    """Initialize a worker for PCGrad.

    This worker samples `num_tasks` trajectories in one rollout in order to
    make sure each task is sampled when optimizing. This worker is always used
    together with `multi_env_wrapper` and `round_robin_strategy`.

    Args:
        seed(int): The seed to use to intialize random number generators.
        max_episode_length(int or float): The maximum length paths which will
            be sampled. Can be (floating point) infinity.
        worker_number(int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
        num_tasks (int): Number of trajectories sampled per rollout.

    Attributes:
        agent(Policy or None): The worker's agent.
        env(Environment or None): The worker's environment.

    """

    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_episode_length,
            worker_number,
            num_tasks=2):
        self._num_tasks = num_tasks
        super().__init__(seed=seed,
                         max_episode_length=max_episode_length,
                         worker_number=worker_number)

    def rollout(self):
        """Sample a single rollout of the agent in the environment.

        Returns:
            garage.TrajectoryBatch: The collected trajectory.

        """
        for _ in range(self._num_tasks):
            self.start_episode()
            while not self.step_episode():
                pass
        return self.collect_episode()
