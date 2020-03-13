"""Module for PEARLWorker."""
from garage import TimeStep
from garage.sampler.worker import DefaultWorker


class PEARLWorker(DefaultWorker):
    """A worker class used in sampling for PEARL.

    It stores context and resample belief in the policy every step.

    Args:
        seed(int): The seed to use to intialize random number generators.
        max_path_length(int or float): The maximum length paths which will
            be sampled. Can be (floating point) infinity.
        worker_number(int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
        deterministic(bool): If true, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.
        accum_context(bool): If true, update context of the agent.

    Attributes:
        agent(Policy or None): The worker's agent.
        env(gym.Env or None): The worker's environment.

    """

    def __init__(self,
                 *,
                 seed,
                 max_path_length,
                 worker_number,
                 deterministic=False,
                 accum_context=False):
        self._deterministic = deterministic
        self._accum_context = accum_context
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)

    def start_rollout(self):
        """Begin a new rollout."""
        self._path_length = 0
        self._prev_obs = self.env.reset()

    def step_rollout(self):
        """Take a single time-step in the current rollout.

        Returns:
            bool: True iff the path is done, either due to the environment
            indicating termination of due to reaching `max_path_length`.

        """
        if self._path_length < self._max_path_length:
            a, agent_info = self.agent.get_action(self._prev_obs)
            if self._deterministic:
                a = agent_info['mean']
            next_o, r, d, env_info = self.env.step(a)
            self._observations.append(self._prev_obs)
            self._rewards.append(r)
            self._actions.append(a)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            for k, v in env_info.items():
                self._env_infos[k].append(v)
            self._path_length += 1
            self._terminals.append(d)
            if self._accum_context:
                s = TimeStep(env_spec=self.env,
                             observation=self._prev_obs,
                             next_observation=next_o,
                             action=a,
                             reward=float(r),
                             terminal=d,
                             env_info=env_info,
                             agent_info=agent_info)
                self.agent.update_context(s)
            if not d:
                self._prev_obs = next_o
                return False
        self._lengths.append(self._path_length)
        self._last_observations.append(self._prev_obs)
        return True

    def rollout(self):
        """Sample a single rollout of the agent in the environment.

        Returns:
            garage.TrajectoryBatch: The collected trajectory.

        """
        self.start_rollout()
        while not self.step_rollout():
            pass
        self._agent_infos['context'] = [self.agent.z.detach().cpu().numpy()
                                        ] * self._max_path_length
        self.agent.sample_from_belief()
        return self.collect_rollout()
