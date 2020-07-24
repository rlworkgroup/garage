"""ϵ-greedy exploration strategy.

Random exploration according to the value of epsilon.
"""
import numpy as np

from garage.np.exploration_policies.exploration_policy import ExplorationPolicy


class EpsilonGreedyPolicy(ExplorationPolicy):
    """ϵ-greedy exploration strategy.

    Select action based on the value of ϵ. ϵ will decrease from
    max_epsilon to min_epsilon within decay_ratio * total_timesteps.

    At state s, with probability
    1 − ϵ: select action = argmax Q(s, a)
    ϵ    : select a random action from an uniform distribution.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        policy (garage.Policy): Policy to wrap.
        total_timesteps (int): Total steps in the training, equivalent to
            max_episode_length * n_epochs.
        max_epsilon (float): The maximum(starting) value of epsilon.
        min_epsilon (float): The minimum(terminal) value of epsilon.
        decay_ratio (float): Fraction of total steps for epsilon decay.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 *,
                 total_timesteps,
                 max_epsilon=1.0,
                 min_epsilon=0.02,
                 decay_ratio=0.1):
        super().__init__(policy)
        self._env_spec = env_spec
        self._max_epsilon = max_epsilon
        self._min_epsilon = min_epsilon
        self._decay_period = int(total_timesteps * decay_ratio)
        self._action_space = env_spec.action_space
        self._epsilon = self._max_epsilon
        self._decrement = (self._max_epsilon -
                           self._min_epsilon) / self._decay_period

    def get_action(self, observation):
        """Get action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from the environment.

        Returns:
            np.ndarray: An action with noise.
            dict: Arbitrary policy state information (agent_info).

        """
        opt_action, _ = self.policy.get_action(observation)
        self._decay()
        if np.random.random() < self._epsilon:
            opt_action = self._action_space.sample()

        return opt_action, dict()

    def get_actions(self, observations):
        """Get actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observation from the environment.

        Returns:
            np.ndarray: Actions with noise.
            List[dict]: Arbitrary policy state information (agent_info).

        """
        opt_actions, _ = self.policy.get_actions(observations)
        for itr, _ in enumerate(opt_actions):
            self._decay()
            if np.random.random() < self._epsilon:
                opt_actions[itr] = self._action_space.sample()

        return opt_actions, dict()

    def _decay(self):
        if self._epsilon > self._min_epsilon:
            self._epsilon -= self._decrement
