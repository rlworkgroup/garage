"""Datatypes used by multiple Samplers or Workers."""
import collections

import numpy as np

from garage import TrajectoryBatch


class InProgressTrajectory:
    """An in-progress trajectory.

    Compared to TrajectoryBatch, this datatype does less checking, only
    contains one trajectory, and uses lists instead of numpy arrays to make
    stepping faster.

    Args:
        env (gym.Env): The environment the trajectory is being collected in.
        initial_observation (np.ndarray): The first observation. If None, the
            environment will be reset to generate this observation.

    """

    def __init__(self, env, initial_observation=None):
        self.env = env
        if initial_observation is None:
            initial_observation = env.reset()
        self.observations = [initial_observation]
        self.actions = []
        self.rewards = []
        self.terminals = []
        self.agent_infos = collections.defaultdict(list)
        self.env_infos = collections.defaultdict(list)

    def step(self, action, agent_info):
        """Step the trajectory using an action from an agent.

        Args:
            action (np.ndarray): The action taken by the agent.
            agent_info (dict[str, np.ndarray]): Extra agent information.

        Returns:
            np.ndarray: The new observation from the environment.

        """
        next_o, r, d, env_info = self.env.step(action)
        self.observations.append(next_o)
        self.rewards.append(r)
        self.actions.append(action)
        for k, v in agent_info.items():
            self.agent_infos[k].append(v)
        for k, v in env_info.items():
            self.env_infos[k].append(v)
        self.terminals.append(d)
        return next_o

    def to_batch(self):
        """Convert this in-progress trajectory into a TrajectoryBatch.

        Returns:
            TrajectoryBatch: This trajectory as a batch.

        Raises:
            AssertionError: If this trajectory contains no time steps.

        """
        assert len(self.rewards) > 0
        return TrajectoryBatch(env_spec=self.env.spec,
                               observations=np.asarray(self.observations[:-1]),
                               last_observations=np.asarray([self.last_obs]),
                               actions=np.asarray(self.actions),
                               rewards=np.asarray(self.rewards),
                               terminals=np.asarray(self.terminals),
                               env_infos=self.env_infos,
                               agent_infos=self.env_infos,
                               lengths=np.asarray([len(self.rewards)],
                                                  dtype='l'))

    @property
    def last_obs(self):
        """np.ndarray: The last observation in the trajectory."""
        return self.observations[-1]
