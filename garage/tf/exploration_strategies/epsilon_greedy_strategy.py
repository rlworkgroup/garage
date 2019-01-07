"""
ϵ-greedy exploration strategy.

Random exploration according to the value of epsilon.
"""
import numpy as np

from garage.exploration_strategies import ExplorationStrategy
from garage.misc.overrides import overrides


class EpsilonGreedyStrategy(ExplorationStrategy):
    """
    ϵ-greedy exploration strategy.

    Select action based on the value of ϵ. ϵ will decrease from
    max_epsilon to min_epsilon within decay_ratio * total_step.

    At state s, with probability
    1 − ϵ: select action = argmax Q(s, a)
    ϵ    : select a random action from an uniform distribution.

    Args:
        env_spec: Environment specification
        total_step: Total steps in the training, max_path_length * n_epochs.
        max_epsilon: The maximum(starting) value of epsilon.
        min_epsilon: The minimum(terminal) value of epsilon.
        decay_ratio: Fraction of total steps for epsilon decay.
    """

    def __init__(self,
                 env_spec,
                 total_step,
                 max_epsilon=1.0,
                 min_epsilon=0.02,
                 decay_ratio=0.1):
        self._env_spec = env_spec
        self._max_epsilon = max_epsilon
        self._min_epsilon = min_epsilon
        self._decay_period = int(total_step * decay_ratio)
        self._action_space = env_spec.action_space
        self.reset()

    @overrides
    def reset(self):
        """Reset the state of the exploration."""
        self._epsilon = self._max_epsilon

    @overrides
    def get_action(self, t, observation, policy, sess=None, **kwargs):
        """
        Get action from this policy for the input observation.

        Args:
            t: Iteration.
            observation: Observation from the environment.
            policy: Policy network to predict action based on the observation.

        Returns:
            opt_action: optimal action from this policy.

        """
        if self._epsilon > self._min_epsilon:
            self._epsilon -= (
                self._max_epsilon - self._min_epsilon) / self._decay_period

        opt_action = policy.get_action(observation, sess)

        if np.random.random() < self._epsilon:
            opt_action = self._action_space.sample()

        return opt_action

    @overrides
    def get_actions(self, t, observations, policy, sess=None, **kwargs):
        """
        Get actions from this policy for the input observations.

        Args:
            t: Iteration.
            observation: Observation from the environment.
            policy: Policy network to predict action based on the observation.

        Returns:
            opt_action: optimal actions from this policy.

        """
        if self._epsilon > self._min_epsilon:
            self._epsilon -= (
                self._max_epsilon - self._min_epsilon) / self._decay_period

        opt_actions = policy.get_actions(observations, sess)
        for itr in range(len(opt_actions)):
            if np.random.random() < self._epsilon:
                opt_actions[itr] = self._action_space.sample()

        return opt_actions
