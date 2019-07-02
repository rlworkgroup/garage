"""
This modules creates a continuous MLP policy network.

A continuous MLP network can be used as policy method in different RL
algorithms. It accepts an observation of the environment and predicts an
action.
"""
import akro
import torch

from garage.misc.overrides import overrides
from garage.tf.policies.base import Policy


class ContinuousMLPPolicy(Policy):
    """
    This class implements a policy network.

    The policy network selects action based on the state of the environment.
    It uses neural nets to fit the function of pi(s).
    """

    def __init__(self,
                 env_spec,
                 nn_module,
                 name='ContinuousMLPPolicy',
                 input_include_goal=False):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec():
            nn_module():
                A PyTorch module.
            name(str, optional):
                A str contains the name of the policy.
        """
        assert isinstance(env_spec.action_space, akro.Box)

        super().__init__(env_spec)

        self._env_spec = env_spec
        self._nn_module = nn_module
        self._name = name
        if input_include_goal:
            self._obs_dim = env_spec.observation_space.flat_dim_with_keys(
                ['observation', 'desired_goal'])
        else:
            self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._action_bound = env_spec.action_space.high

    def forward(self, input_val):
        """Forward method."""
        x = torch.FloatTensor(self._nn_module(input_val))
        return torch.distributions.Uniform(x, x)

    @overrides
    def get_action(self, observation):
        """Return a single action."""
        with torch.no_grad():
            d = self.forward(observation.unsqueeze(0))
            return d.sample().detach().numpy()

    @overrides
    def get_actions(self, observations):
        """Return multiple actions."""
        with torch.no_grad():
            d = self.forward(observations)
            return d.sample().detach().numpy()
