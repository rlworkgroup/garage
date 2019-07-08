"""
This modules creates a continuous policy network.

A continuous neural network can be used as policy method in different RL
algorithms. It accepts an observation of the environment and predicts an
action.
"""
import akro
import torch

from garage.tf.policies.base import Policy


class ContinuousNNPolicy(Policy):
    """
    Implements a module-agnostic policy network.

    The policy network selects action based on the state of the environment.
    It uses a PyTorch neural network module to fit the function of pi(s).
    """

    def __init__(self,
                 env_spec,
                 nn_module,
                 policy_type='Deterministic',
                 input_include_goal=False):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            nn_module (nn.Module): Neural network module in PyTorch.
            policy_type (str): Type of policy, deterministic or stochastic.
        """
        assert isinstance(env_spec.action_space, akro.Box)

        super().__init__(env_spec)

        self._env_spec = env_spec
        self._nn_module = nn_module
        self._policy_type = policy_type
        if input_include_goal:
            self._obs_dim = env_spec.observation_space.flat_dim_with_keys(
                ['observation', 'desired_goal'])
        else:
            self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._action_bound = env_spec.action_space.high

    def forward(self, input_val):
        """Forward method."""
        return self._nn_module(input_val)

    def get_action(self, observation):
        """Return a single action."""
        with torch.no_grad():
            x = self.forward(observation.unsqueeze(0))
            if self._policy_type == 'Deterministic':
                return x.detach().numpy().squeeze(0)
            else:
                return x.sample().detach().numpy().squeeze(0)

    def get_actions(self, observations):
        """Return multiple actions."""
        with torch.no_grad():
            x = self.forward(observations)
            if self._policy_type == 'Deterministic':
                return x.detach().numpy()
            else:
                return x.sample().detach().numpy()
