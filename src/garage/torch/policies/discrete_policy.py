"""Base Discrete Policy."""
import abc

import akro
import torch

from garage.torch import global_device
from garage.torch.policies.policy import Policy


class DiscretePolicy(Policy, abc.ABC):
    """Abstract base class for torch discrete policies."""

    def get_action(self, observation):
        r"""Get a single action given an observation.

        Args:
            observation (np.ndarray): Observation from the environment.
                Shape is :math:`env_spec.observation_space`.

        Returns:
            np.ndarray[int]: Predicted categorical action given
                input observation. Shape is :math:`env_spec.action_space`.
        """
        with torch.no_grad():
            if not isinstance(observation, torch.Tensor):
                observation = torch.as_tensor(observation).float().to(
                    global_device())
            if isinstance(self._env_spec.observation_space, akro.Image):
                observation /= 255.0  # scale image
            observation = observation.unsqueeze(0)
            dist = self.forward(observation)
            return dist.sample().squeeze(0).cpu().numpy()

    def get_actions(self, observations):
        r"""Get actions given observations.

        Args:
            observations (np.ndarray): Observations from the environment.
                Shape is :math:`batch_dim \bullet env_spec.observation_space`.

        Returns:
            np.ndarray[int]: Predicted categorical action given
                 input observation.
                 :math:`batch_dim \bullet env_spec.action_space`.

        """
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(
                    global_device())
            if isinstance(self._env_spec.observation_space, akro.Image):
                observations /= 255.0  # scale image
            dist = self.forward(observations)
            return dist.sample().cpu().numpy()

    # pylint: disable=arguments-differ
    @abc.abstractmethod
    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors.
                Do not need to be detached, and can be on any device.
        """
