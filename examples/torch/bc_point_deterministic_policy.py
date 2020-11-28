#!/usr/bin/env python3
"""Example of using Behavioral Cloning."""
import numpy as np

from garage import wrap_experiment
from garage.envs import PointEnv
from garage.sampler import RaySampler
from garage.torch.algos import BC
from garage.torch.policies import DeterministicMLPPolicy, Policy
from garage.trainer import Trainer


class OptimalPolicy(Policy):
    """Optimal policy for PointEnv.

    Args:
        env_spec (EnvSpec): The environment spec.
        goal (np.ndarray): The goal location of the environment.

    """

    # No forward method
    # pylint: disable=abstract-method

    def __init__(self, env_spec, goal):
        super().__init__(env_spec, 'OptimalPolicy')
        self.goal = goal

    def get_action(self, observation):
        """Get action given observation.

        Args:
            observation (np.ndarray): Observation from PointEnv. Should have
                length at least 2.

        Returns:
            tuple:
                * np.ndarray: Optimal action in the environment. Has length 2.
                * dict[str, np.ndarray]: Agent info (empty).

        """
        return self.goal - observation[:2], {}

    def get_actions(self, observations):
        """Get actions given observations.

        Args:
            observations (np.ndarray): Observations from the environment.
                Has shape :math:`(B, O)`, where :math:`B` is the batch
                dimension and :math:`O` is the observation dimensionality (at
                least 2).

        Returns:
            tuple:
                * np.ndarray: Batch of optimal actions.
                    Has shape :math:`(B, 2)`, where :math:`B` is the batch
                    dimension.
                Optimal action in the environment.
                * dict[str, np.ndarray]: Agent info (empty).

        """
        return (self.goal[np.newaxis, :].repeat(len(observations), axis=0) -
                observations[:, :2]), {}


@wrap_experiment
def bc_point(ctxt=None):
    """Run Behavioral Cloning on garage.envs.PointEnv.

    Args:
        ctxt (ExperimentContext): Provided by wrap_experiment.

    """
    trainer = Trainer(ctxt)
    goal = np.array([1., 1.])
    env = PointEnv(goal=goal, max_episode_length=200)
    expert = OptimalPolicy(env.spec, goal=goal)
    policy = DeterministicMLPPolicy(env.spec, hidden_sizes=[8, 8])
    batch_size = 1000
    sampler = RaySampler(agents=expert,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)
    algo = BC(env.spec,
              policy,
              batch_size=batch_size,
              source=expert,
              sampler=sampler,
              policy_lr=1e-2,
              loss='mse')
    trainer.setup(algo, env)
    trainer.train(100, batch_size=batch_size)


bc_point()
