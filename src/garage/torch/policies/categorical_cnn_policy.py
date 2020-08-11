"""CategoricalCNNPolicy."""
import akro
import torch
from torch import nn

from garage.torch.modules import CategoricalCNNModule
from garage.torch.policies.stochastic_policy import StochasticPolicy


class CategoricalCNNPolicy(StochasticPolicy):
    """CategoricalCNNPolicy.

    A policy that contains a CNN and a MLP to make prediction based on
    a categorical distribution.

    It only works with akro.Discrete action space.

    Args:
        env (garage.envs): Environment.
        kernel_sizes (tuple[int]): Dimension of the conv filters.
            For example, (3, 5) means there are two convolutional layers.
            The filter for first layer is of dimension (3 x 3)
            and the second one is of dimension (5 x 5).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_channels (tuple[int]): Number of output channels for CNN.
            For example, (3, 32) means there are two convolutional layers.
            The filter for the first conv layer outputs 3 channels
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        paddings (tuple[int]):  Zero-padding added to both sides of the input
        padding_mode (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        max_pool (bool): Bool for using max-pooling or not.
        pool_shape (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            shape (2, 2).
        pool_stride (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.

    """

    def __init__(self,
                 env,
                 kernel_sizes,
                 hidden_channels,
                 strides=1,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 paddings=0,
                 padding_mode='zeros',
                 max_pool=False,
                 pool_shape=None,
                 pool_stride=1,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='CategoricalCNNPolicy'):

        if not isinstance(env.spec.action_space, akro.Discrete):
            raise ValueError('CategoricalMLPPolicy only works '
                             'with akro.Discrete action space.')
        if isinstance(env.spec.observation_space, akro.Dict):
            raise ValueError('CNN policies do not support '
                             'with akro.Dict observation spaces.')

        super().__init__(env.spec, name)
        self._env = env
        self._obs_dim = self._env.spec.observation_space.shape
        self._action_dim = self._env.spec.action_space.flat_dim
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_conv_channels = hidden_channels
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._hidden_sizes = hidden_sizes
        self._paddings = paddings
        self._padding_mode = padding_mode
        self._max_pool = max_pool
        self._pool_shape = pool_shape
        self._pool_stride = pool_stride
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        self._is_image = isinstance(self._env.spec.observation_space,
                                    akro.Image)

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
        module = CategoricalCNNModule(
            input_var=observations,
            output_dim=self._action_dim,
            kernel_sizes=self._kernel_sizes,
            strides=self._strides,
            hidden_channels=self._hidden_conv_channels,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            paddings=self._paddings,
            padding_mode=self._padding_mode,
            max_pool=self._max_pool,
            pool_shape=self._pool_shape,
            pool_stride=self._pool_stride,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization,
            is_image=self._is_image)

        dist = module(observations)
        return dist, {}
