"""This modules creates a deterministic policy network.

A neural network can be used as policy method in different RL algorithms.
It accepts an observation of the environment and predicts an action.
"""
import torch

from garage.torch.modules import MLPModule
from garage.torch.policies.policy import Policy


class DeterministicMLPPolicy(Policy):
    """Implements a deterministic policy network.

    The policy network selects action based on the state of the environment.
    It uses a PyTorch neural network module to fit the function of pi(s).
    """

    def __init__(self, env_spec, name='DeterministicMLPPolicy', **kwargs):
        """Initialize class with multiple attributes.

        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            name (str): Policy name.
            kwargs : Additional keyword arguments passed to the MLPModule.
        """
        super().__init__(env_spec, name)

        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._module = MLPModule(input_dim=self._obs_dim,
                                 output_dim=self._action_dim,
                                 **kwargs)

    # pylint: disable=arguments-differ
    def forward(self, observations):
        """Compute actions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.Tensor: Batch of actions.
        """
        return self._module(observations)

    def get_action(self, observation):
        """Get a single action given an observation.

        Args:
            observation (np.ndarray): Observation from the environment.

        Returns:
            tuple:
                * np.ndarray: Predicted action.
                * dict:
                    * np.ndarray[float]: Mean of the distribution
                    * np.ndarray[float]: Log of standard deviation of the
                        distribution
        """
        with torch.no_grad():
            x = self(torch.Tensor(observation).unsqueeze(0))
            return x.squeeze(0).numpy(), dict()

    def get_actions(self, observations):
        """Get actions given observations.

        Args:
            observations (np.ndarray): Observations from the environment.

        Returns:
            tuple:
                * np.ndarray: Predicted actions.
                * dict:
                    * np.ndarray[float]: Mean of the distribution
                    * np.ndarray[float]: Log of standard deviation of the
                        distribution
        """
        with torch.no_grad():
            x = self(torch.Tensor(observations))
            return x.numpy(), dict()
