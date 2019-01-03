"""
ϵ-greedy policy.

Random exploration according to the value of epsilon.
"""
import numpy as np

from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.tf.policies import Policy
from garage.tf.spaces import Box


class EpsilonGreedyPolicy(Policy, Serializable):
    """
    ϵ-greedy policy.

    Select action based on the value of ϵ. ϵ will decrease from
    max_epsilon to min_epsilon within decay_ratio * total_step.

    At state s, with probability
    1 − ϵ: select action = argmax Q(s, a)
    ϵ    : select a random action

    Args:
        env_spec: environment specification
        total_step: Total steps in the training.
        max_epsilon: the maximum(starting) value of epsilon.
        min_epsilon: the minimum(terminal) value of epsilon.
        decay_ratio: Fraction of total steps for epsilon decay.
    """

    def __init__(self,
                 env_spec,
                 total_step,
                 max_epsilon=1.0,
                 min_epsilon=0.02,
                 decay_ratio=0.1):
        Serializable.quick_init(self, locals())
        super(EpsilonGreedyPolicy, self).__init__(env_spec)

        self._env_spec = env_spec
        self._max_epsilon = max_epsilon
        self._epsilon = max_epsilon
        self._min_epsilon = min_epsilon
        self._decay_period = int(total_step * decay_ratio)
        self._action_space = env_spec.action_space
        self._act = None

    def set_act(self, act):
        """
        Set the act function.

        The act function is used to choose action
        given observations.
        """
        self._act = act

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True

    @overrides
    def get_action(self, observation):
        """
        Get action from this policy for the input observation.

        Args:
            observation: Observation from environment.

        Returns:
            opt_action: optimal action from this policy.

        """
        assert self._act is not None

        if self._epsilon > self._min_epsilon:
            self._epsilon -= (
                self._max_epsilon - self._min_epsilon) / self._decay_period

        opt_action = np.argmax(self._act(observation))

        if np.random.random() < self._epsilon:
            opt_action = np.random.randint(0, self._action_space.n)

        return opt_action, dict()

    @overrides
    def get_actions(self, observations):
        """
        Get actions from this policy for the input observations.

        Args:
            observations: Observations from environment.

        Returns:
            opt_action: optimal actions from this policy.

        """
        assert self._act is not None

        if self._epsilon > self._min_epsilon:
            self._epsilon -= (
                self._max_epsilon - self._min_epsilon) / self._decay_period

        opt_action = np.argmax(self._act(observations), axis=1)

        for itr in range(len(opt_action)):
            if np.random.random() < self._epsilon:
                if isinstance(self._action_space, Box):
                    opt_action[itr] = np.random.randint(
                        0, self._action_space.flat_dim)
                else:
                    opt_action[itr] = np.random.randint(
                        0, self._action_space.n)

        return opt_action, dict()
