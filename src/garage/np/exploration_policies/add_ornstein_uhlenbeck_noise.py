"""Ornstein-Uhlenbeck exploration strategy.

Ornstein-Uhlenbeck exploration strategy comes from the Ornstein-Uhlenbeck
process. It is often used in DDPG algorithm because in continuous control task
it is better to have temporally correlated exploration to get smoother
transitions. And OU process is relatively smooth in time.
"""
import numpy as np

from garage.np.exploration_policies.exploration_policy import ExplorationPolicy


class AddOrnsteinUhlenbeckNoise(ExplorationPolicy):
    r"""An exploration strategy based on the Ornstein-Uhlenbeck process.

    The process is governed by the following stochastic differential equation.

    .. math::
       dx_t = -\theta(\mu - x_t)dt + \sigma \sqrt{dt} \mathcal{N}(\mathbb{0}, \mathbb{1})  # noqa: E501

    Args:
        env_spec (EnvSpec): Environment to explore.
        policy (garage.Policy): Policy to wrap.
        mu (float): :math:`\mu` parameter of this OU process. This is the drift
            component.
        sigma (float): :math:`\sigma > 0` parameter of this OU process. This is
            the coefficient for the Wiener process component. Must be greater
            than zero.
        theta (float): :math:`\theta > 0` parameter of this OU process. Must be
            greater than zero.
        dt (float): Time-step quantum :math:`dt > 0` of this OU process. Must
            be greater than zero.
        x0 (float): Initial state :math:`x_0` of this OU process.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 *,
                 mu=0,
                 sigma=0.3,
                 theta=0.15,
                 dt=1e-2,
                 x0=None):
        super().__init__(policy)
        self._env_spec = env_spec
        self._action_space = env_spec.action_space
        self._action_dim = self._action_space.flat_dim
        self._mu = mu
        self._sigma = sigma
        self._theta = theta
        self._dt = dt
        self._x0 = x0 if x0 is not None else self._mu * np.zeros(
            self._action_dim)
        self._state = self._x0

    def _simulate(self):
        """Advance the OU process.

        Returns:
            np.ndarray: Updated OU process state.

        """
        x = self._state
        dx = self._theta * (self._mu - x) * self._dt + self._sigma * np.sqrt(
            self._dt) * np.random.normal(size=len(x))
        self._state = x + dx
        return self._state

    def reset(self, dones=None):
        """Reset the state of the exploration.

        Args:
            dones (List[bool] or numpy.ndarray or None): Which vectorization
                states to reset.

        """
        self._state = self._x0
        super().reset(dones)

    def get_action(self, observation):
        """Return an action with noise.

        Args:
            observation (np.ndarray): Observation from the environment.

        Returns:
            np.ndarray: An action with noise.
            dict: Arbitrary policy state information (agent_info).

        """
        action, agent_infos = self.policy.get_action(observation)
        ou_state = self._simulate()
        return np.clip(action + ou_state, self._action_space.low,
                       self._action_space.high), agent_infos

    def get_actions(self, observations):
        """Return actions with noise.

        Args:
            observations (np.ndarray): Observation from the environment.

        Returns:
            np.ndarray: Actions with noise.
            List[dict]: Arbitrary policy state information (agent_info).

        """
        actions, agent_infos = self.policy.get_actions(observations)
        ou_state = self._simulate()
        return np.clip(actions + ou_state, self._action_space.low,
                       self._action_space.high), agent_infos
