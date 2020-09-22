"""ϵ-greedy exploration strategy.

Random exploration according to the value of epsilon.
"""
from dowel import tabular
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
        self._decrement = (self._max_epsilon -
                           self._min_epsilon) / self._decay_period
        self._total_env_steps = 0
        self._last_total_env_steps = 0

    @property
    def epsilon(self):
        """Float: the instantaneous level of exploration noise."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        self._episilon = epsilon

    def get_action(self, observation):
        """Get action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from the environment.

        Returns:
            np.ndarray: An action with noise.
            dict: Arbitrary policy state information (agent_info).

        """
        opt_action, _ = self.policy.get_action(observation)
        if np.random.random() < self._epsilon():
            opt_action = self._action_space.sample()
        self._total_env_steps += 1
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
            if np.random.random() < self._epsilon():
                opt_actions[itr] = self._action_space.sample()
            self._total_env_steps += 1

        return opt_actions, dict()

    def _epsilon(self):
        """Get the current epsilon.

        Returns:
            double: Epsilon.

        """
        if self._total_env_steps >= self._decay_period:
            return self._min_epsilon
        return self._max_epsilon - self._decrement * self._total_env_steps

    def update(self, episode_batch):
        """Update the exploration policy using a batch of trajectories.

        Args:
            episode_batch (EpisodeBatch): A batch of trajectories which
                were sampled with this policy active.

        """
        self._total_env_steps = (self._last_total_env_steps +
                                 np.sum(episode_batch.lengths))
        self._last_total_env_steps = self._total_env_steps
        tabular.record('EpsilonGreedyPolicy/Epsilon', self._epsilon())

    def get_param_values(self):
        """Get parameter values.

        Returns:
            list or dict: Values of each parameter.

        """
        return {
            'total_env_steps': self._total_env_steps,
            'inner_params': self.policy.get_param_values()
        }

    def set_param_values(self, params):
        """Set param values.

        Args:
            params (np.ndarray): A numpy array of parameter values.

        """
        self._total_env_steps = params['total_env_steps']
        self.policy.set_param_values(params['inner_params'])
        self._last_total_env_steps = self._total_env_steps
