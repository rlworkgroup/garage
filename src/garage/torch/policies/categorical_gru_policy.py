"""CategoricalGRUPolicy."""
import akro
import numpy as np
import torch
from torch import nn

from garage.torch.modules import CategoricalGRUModule
from garage.torch.policies.stochastic_policy import StochasticPolicy


class CategoricalGRUPolicy(StochasticPolicy):
    """CategoricalGRUPolicy.

    A policy that contains a GRU and a MLP to make prediction based on
    a categorical distribution.

    It only works with akro.Discrete action space.

    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_dim (int): Hidden dimension for GRU cell.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        state_include_action (bool): Whether the state includes action.
            If True, input dimension will be
            (observation dimension + action dimension).
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.
    """

    def __init__(
        self,
        env_spec,
        hidden_dim=32,
        hidden_nonlinearity=nn.Tanh,
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        output_nonlinearity=None,
        output_w_init=nn.init.xavier_uniform_,
        output_b_init=nn.init.zeros_,
        state_include_action=True,
        layer_normalization=False,
        name="CategoricalGRUPolicy",
    ):
        if not isinstance(env_spec.action_space, akro.Discrete):
            raise ValueError('CategoricalGRUPolicy only works'
                             'with akro.Discrete action space.')

        super().__init__(env_spec, name)
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.n

        self._hidden_dim = hidden_dim
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        self._state_include_action = state_include_action

        if state_include_action:
            self._input_dim = self._obs_dim + self._action_dim
        else:
            self._input_dim = self._obs_dim
        
        self._prev_actions = None

        self._module = CategoricalGRUModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_dim=self._hidden_dim,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization,
        )

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
        dist = self._module(observations)
        return dist, {}

    def reset(self, do_resets=None):
        """Reset the policy.

        Note:
            If `do_resets` is None, it will be by default np.array([True]),
            which implies the policy will not be "vectorized", i.e. number of
            paralle environments for training data sampling = 1.

        Args:
            do_resets (numpy.ndarray): Bool that indicates terminal state(s).

        """
        if do_resets is None:
            do_resets = [True]
        do_resets = np.asarray(do_resets)
        if self._prev_actions is None or len(do_resets) != len(
                self._prev_actions):
            self._prev_actions = np.zeros(
                (len(do_resets), self.action_space.flat_dim))
            self._prev_hiddens = np.zeros((len(do_resets), self._hidden_dim))

        self._prev_actions[do_resets] = 0.
    
    def get_actions(self, observations):
        """Return multiple actions.

        Args:
            observations (numpy.ndarray): Observations.

        Returns:
            list[int]: Actions given input observations.
            dict(numpy.ndarray): Distribution parameters.

        """
        if self._state_include_action:
            assert self._prev_actions is not None
            all_input = np.concatenate([observations, self._prev_actions],
                                       axis=-1)
        else:
            all_input = observations
        prev_actions = self._prev_actions
        actions, agent_info = super().get_actions(all_input)
        self._prev_actions = self.action_space.flatten_n([a.item() for a in actions])
        if self._state_include_action:
            agent_info['prev_action'] = np.copy(prev_actions)
        return actions, agent_info
    
    @property
    def input_dim(self):
        """int: Dimension of the policy input."""
        return self._input_dim
    
    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """
        return self._env_spec
    
    
