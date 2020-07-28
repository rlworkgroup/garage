import numpy as np
from dowel import tabular

from garage.np.exploration_policies import ExplorationPolicy


class AddExpertActions(ExplorationPolicy):
    """ExplorationPolicy that queries an expert for actions with probability p.

    """

    def __init__(self, env_spec, policy, expert_policy, initial_expert_p=1.0,
                 final_expert_p=0.1, decay_period=int(1e6)):
        super().__init__(policy)
        self.env_spec = env_spec
        self.expert_policy = expert_policy
        self.initial_expert_p = initial_expert_p
        self.final_expert_p = final_expert_p
        self.decay_period = decay_period
        self.total_steps_so_far = 0
        self.current_expert_p = initial_expert_p

    def get_action(self, observation):
        """Return an action with noise.

        Args:
            observation (np.ndarray): Observation from the environment.

        Returns:
            np.ndarray: An action with noise.
            dict: Arbitrary policy state information (agent_info).

        """
        policy_action, policy_info = self.policy.get_action(observation)
        assert 'mean' in policy_info
        if np.random.uniform() < self.current_expert_p:
            expert_action = self.expert_policy.get_action(observation)
            low = getattr(self.env_spec.action_space, 'low')
            high = getattr(self.env_spec.action_space, 'high')
            if low is not None and high is not None:
                expert_action = np.clip(expert_action, low, high)
            return expert_action, policy_info
        else:
            return policy_action, policy_info

    def get_actions(self, observations):
        """Return actions with noise.

        Args:
            observations (np.ndarray): Observation from the environment.

        Returns:
            np.ndarray: Actions with noise.
            List[dict]: Arbitrary policy state information (agent_info).

        """
        actions, info = self.policy.get_actions(observations)
        for i, obs in enumerate(observations):
            if np.random.uniform() < self.current_expert_p:
                expert_action = self.expert_policy.get_action(obs)
                low = getattr(self.env_spec.action_space, 'low')
                high = getattr(self.env_spec.action_space, 'high')
                if low is not None and high is not None:
                    expert_action = np.clip(expert_action, low, high)
                actions[i] = expert_action
        return actions, info

    def update(self, trajectory_batch):
        """Update the exploration policy using a batch of trajectories.

        Args:
            trajectory_batch (TrajectoryBatch): A batch of trajectories which
                were sampled with this policy active.

        """
        self.total_steps_so_far += np.sum(trajectory_batch.lengths)
        if self.total_steps_so_far >= self.decay_period:
            self.current_expert_p = self.final_expert_p
        else:
            self.current_expert_p = np.interp(x=[self.total_steps_so_far],
                                              xp=[0, self.decay_period],
                                              fp=[self.initial_expert_p,
                                                  self.final_expert_p])[0]

        tabular.record('AddExpertActions/TotalEnvSteps',
                       self.total_steps_so_far)
        tabular.record('AddExpertActions/CurrentExpertProbability',
                       self.current_expert_p)

    def get_param_values(self):
        """Get parameter values.

        Returns:
            list or dict: Values of each parameter.

        """
        return {'total_steps_so_far': self.total_steps_so_far,
                'inner_params': self.policy.get_param_values()}

    def set_param_values(self, params):
        """Set param values.

        Args:
            params (np.ndarray): A numpy array of parameter values.

        """
        self.total_steps_so_far = params['total_steps_so_far']
        self.policy.set_param_values(params['inner_params'])

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def __call__(self, *args, **kwargs):
        return self.policy(*args, **kwargs)
