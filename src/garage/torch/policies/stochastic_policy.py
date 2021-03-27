"""Base Stochastic Policy."""
import abc

import akro
import numpy as np
import torch

from garage.torch._functions import list_to_tensor, np_to_torch
from garage.torch.policies.policy import Policy


class StochasticPolicy(Policy, abc.ABC):
    """Abstract base class for torch stochastic policies."""

    def get_action(self, observation):
        r"""Get a single action given an observation.

        Args:
            observation (np.ndarray): Observation from the environment.
                Shape is :math:`env_spec.observation_space`.

        Returns:
            tuple:
                * np.ndarray: Predicted action. Shape is
                    :math:`env_spec.action_space`.
                * dict:
                    * np.ndarray[float]: Mean of the distribution
                    * np.ndarray[float]: Standard deviation of logarithmic
                        values of the distribution.
        """
        if not isinstance(observation, np.ndarray) and not isinstance(
                observation, torch.Tensor):
            observation = self._env_spec.observation_space.flatten(observation)
        elif isinstance(observation,
                        np.ndarray) and len(observation.shape) > 1:
            observation = self._env_spec.observation_space.flatten(observation)
        elif isinstance(observation,
                        torch.Tensor) and len(observation.shape) > 1:
            observation = torch.flatten(observation)
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                observation = np_to_torch(observation)
            if not isinstance(observation, torch.Tensor):

                observation = list_to_tensor(observation)
            observation = observation.unsqueeze(0)
            action, agent_infos = self.get_actions(observation)
            return action[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        r"""Get actions given observations.

        Args:
            observations (np.ndarray): Observations from the environment.
                Shape is :math:`batch_dim \bullet env_spec.observation_space`.

        Returns:
            tuple:
                * np.ndarray: Predicted actions.
                    :math:`batch_dim \bullet env_spec.action_space`.
                * dict:
                    * np.ndarray[float]: Mean of the distribution.
                    * np.ndarray[float]: Standard deviation of logarithmic
                        values of the distribution.
        """
        if not isinstance(observations[0], np.ndarray) and not isinstance(
                observations[0], torch.Tensor):
            observations = self._env_spec.observation_space.flatten_n(
                observations)

        # frequently users like to pass lists of torch tensors or lists of
        # numpy arrays. This handles those conversions.
        if isinstance(observations, list):
            if isinstance(observations[0], np.ndarray):
                observations = np.stack(observations)
            elif isinstance(observations[0], torch.Tensor):
                observations = torch.stack(observations)

        if isinstance(observations[0],
                      np.ndarray) and len(observations[0].shape) > 1:
            observations = self._env_spec.observation_space.flatten_n(
                observations)
        elif isinstance(observations[0],
                        torch.Tensor) and len(observations[0].shape) > 1:
            observations = torch.flatten(observations, start_dim=1)
        with torch.no_grad():
            if isinstance(observations, np.ndarray):
                observations = np_to_torch(observations)
            if not isinstance(observations, torch.Tensor):

                observations = list_to_tensor(observations)

            if isinstance(self._env_spec.observation_space, akro.Image):
                observations /= 255.0  # scale image
            dist, info = self.forward(observations)
            return dist.sample().cpu().numpy(), {
                k: v.detach().cpu().numpy()
                for (k, v) in info.items()
            }

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
