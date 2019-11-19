"""Gaussian exploration strategy."""
import gym
import numpy as np

from garage.np.exploration_strategies.base import ExplorationStrategy


class GaussianStrategy(ExplorationStrategy):
    """Add Gaussian noise to the action taken by the deterministic policy."""

    def __init__(self,
                 env_spec,
                 max_sigma=1.0,
                 min_sigma=0.1,
                 decay_period=1000000):
        assert isinstance(env_spec.action_space, gym.spaces.Box)
        assert len(env_spec.action_space.shape) == 1
        self._max_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self._action_space = env_spec.action_space

    def get_action(self, iteration, observation, policy, **kwargs):
        """
        Get action from this policy for the input observation.

        Args:
            iteration(int): Iteration.
            observation(numpy.ndarray): Observation from the environment.
            policy(garage.tf.policies.base.Policy):
                Policy network to predict action based on the observation.

        Returns:
            opt_action(numpy.ndarray): optimal action from this policy.
            agent_info(dict): Agent information.

        """
        action, agent_info = policy.get_action(observation)
        sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * min(
            1.0, iteration * 1.0 / self._decay_period)
        return np.clip(action + np.random.normal(size=len(action)) * sigma,
                       self._action_space.low,
                       self._action_space.high), agent_info

    def get_actions(self, iteration, observations, policy, **kwargs):
        """
        Get actions from this policy for the input observation.

        Args:
            iteration(int): Iteration.
            observatioan(list): Observationa from the environment.
            policy(garage.tf.policies.base.Policy):
                Policy network to predict action based on the observation.

        Returns:
            opt_actions(numpy.ndarray): optimal actions from this policy.
            agent_infos(dict): Agent information.

        """
        actions, agent_infos = policy.get_actions(observations)
        sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * min(
            1.0, iteration * 1.0 / self._decay_period)
        return np.clip(actions + np.random.normal(size=len(actions)) * sigma,
                       self._action_space.low,
                       self._action_space.high), agent_infos
