"""GaussianLSTMPolicy."""
import torch
from torch import nn
from torch.distributions import Normal

from garage.torch import (observation_batch_to_packed_sequence,
                          ObservationBatch, ObservationOrder,
                          ShuffledOptimizationNotSupported)
from garage.torch.modules import GaussianMLPModule
from garage.torch.policies.stochastic_policy import StochasticPolicy


class GaussianLSTMPolicy(StochasticPolicy):
    """LSTM whose outputs are fed into a Normal distribution..

    A policy that contains a LSTM to make prediction based on a gaussian
    distribution.

    Args:
        env_spec (EnvSpec): Environment specification.
        name (str): Name of policy.
        hidden_size (int): Hidden dimension for LSTM cell for mean.
        num_layers (int): Number of recurrent layers.
        hidden_mlp_nonlinearity (Callable): Activation function for
            intermediate dense layer(s). It should return a torch.Tensor. Set
            it to None to maintain a linear activation.
        hidden_mlp_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_mlp_w_init (Callable): Initializer function for the weight
            of intermediate dense layer(s). Should modify a torch.Tensor.
        hidden_mlp_b_init (Callable): Initializer function for the bias
            of intermediate dense layer(s). Should modify a torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (Callable): Initializer function for the weight
            of output dense layer(s). Should modify a torch.Tensor.
        output_b_init (Callable): Initializer function for the bias
            of output dense layer(s). Should modify a torch.Tensor.
        recurrent_w_init (Callable): Initializer function for the weight
            of recurrent layer(s). Should modify a torch.Tensor.
        hidden_state_init (Callable): Initializer function for the
            initial hidden state. Should modify a torch.Tensor.
        hidden_state_init_trainable (bool): Bool for whether the initial
            hidden state is trainable.
        cell_state_init (Callable): Initializer function for the
            initial cell state. Should modify a torch.Tensor.
        cell_state_init_trainable (bool): Bool for whether the initial
            cell state is trainable.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
        min_std (float): Minimum value for std.
        max_std (float): Maximum value for std.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation.
            - softplus: the std will be computed as log(1+exp(x)).
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self,
                 env_spec,
                 *,
                 name='GaussianLSTMPolicy',
                 hidden_size=32,
                 num_layers=2,
                 cell_state_init_trainable=True,
                 cell_state_init=None,
                 hidden_state_init=None,
                 hidden_state_init_trainable=True,
                 recurrent_w_init=None,
                 hidden_mlp_sizes=(32, ),
                 hidden_mlp_w_init=nn.init.xavier_uniform_,
                 hidden_mlp_b_init=nn.init.zeros_,
                 hidden_mlp_nonlinearity=torch.tanh,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal):
        super().__init__(env_spec, name)

        if std_parameterization not in ('exp', 'softplus'):
            raise NotImplementedError

        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        h0 = torch.zeros(num_layers, hidden_size)
        c0 = torch.zeros(num_layers, hidden_size)
        if cell_state_init is not None:
            hidden_state_init(h0)
            cell_state_init(c0)
        if cell_state_init_trainable:
            self._c0 = nn.Parameter(c0)
        else:
            self._c0 = c0
            self.register_buffer('_c0', self._c0)
        if hidden_state_init_trainable:
            self._h0 = nn.Parameter(h0)
        else:
            self._h0 = h0
            self.register_buffer('_h0', self._h0)
        self._rnn = nn.LSTM(input_size=self._obs_dim,
                            hidden_size=hidden_size,
                            batch_first=False,
                            num_layers=num_layers)
        if recurrent_w_init is not None:
            recurrent_w_init(self._rnn)
        self._mlp = GaussianMLPModule(
            input_dim=hidden_size,
            output_dim=self._action_dim,
            hidden_sizes=hidden_mlp_sizes,
            hidden_nonlinearity=hidden_mlp_nonlinearity,
            hidden_w_init=hidden_mlp_w_init,
            hidden_b_init=hidden_mlp_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization,
            normal_distribution_cls=normal_distribution_cls)
        self._state = None

    def _new_state(self, n_envs):
        """Compute a new state for running n_envs in parallel.

        Args:
            n_envs (int): Number of observations and actions each call to
                get_actions().

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tensor containing h0 and c0 with
                "batch_dim" repeated n_envs times.

        """
        h0 = self._h0.unsqueeze(1).expand(
            (self._num_layers, n_envs, self._hidden_size))
        c0 = self._c0.unsqueeze(1).expand(
            (self._num_layers, n_envs, self._hidden_size))
        return h0, c0

    def reset(self, do_resets=None):
        """Reset the policy.

        This is effective only to recurrent policies.

        do_resets is an array of boolean indicating
        which internal states to be reset. The length of do_resets should be
        equal to the length of inputs, i.e. batch size.

        Args:
            do_resets (numpy.ndarray): Bool array indicating which states
                to be reset.

        """
        if do_resets is None:
            do_resets = [True]
        h0, c0 = self._new_state(len(do_resets))
        if all(do_resets):
            self._state = (h0, c0)
        for i, do_reset in enumerate(do_resets):
            if do_reset:
                # Reset all layer's state
                self._state[0][:, i] = h0[:, i]
                self._state[1][:, i] = c0[:, i]

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Raises:
            ValueError: If observations is not consistent with reset().
            ShuffledOptimizationNotSupported: If passed a shuffled
                ObservationBatch or a tensor that is not an ObservationBatch.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """
        if not isinstance(observations, ObservationBatch):
            raise ShuffledOptimizationNotSupported(
                f'observations are of type {type(observations)!r}, but should '
                f'be an ObservationBatch')
        if observations.order == ObservationOrder.LAST:
            if self._state is None:
                raise ValueError('get_action() called before reset()')
            if self._state[0].shape[1] != len(observations):
                raise ValueError(f'observations has length '
                                 f'{len(observations)} but should have length '
                                 f'{len(self._state[0])} to match the length '
                                 f'of do_resets in reset()')
            # Add sequence dimension.
            rnn_out, self._state = self._rnn(observations.unsqueeze(0),
                                             self._state)
        else:
            sequence = observation_batch_to_packed_sequence(observations)
            n_episodes = len(observations.lengths)
            start = self._new_state(n_episodes)
            rnn_out, _ = self._rnn(sequence, start)
        # Remove sequence dimension.
        dist = self._mlp(rnn_out.squeeze(0))
        return (dist, dict(mean=dist.mean, log_std=(dist.variance**.5).log()))
