"""Discrete CNN Policy."""
import akro
import torch
from torch import nn

from garage.torch.modules import DiscreteCNNModule
from garage.torch.policies.stochastic_policy import StochasticPolicy


class DiscreteCNNPolicy(StochasticPolicy):
    """DiscreteCNNPolicy.

    A policy that contains a CNN and a MLP to make prediction based on
    a discrete distribution.

    Args:
        env_spec (EnvSpec): Environment specification.
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
        mlp_hidden_nonlinearity (callable): Activation function for
            intermediate dense layer(s) in the MLP. It should return
            a torch.Tensor. Set it to None to maintain a linear activation.
        cnn_hidden_nonlinearity (callable): Activation function for
            intermediate CNN layer(s). It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
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
                 env_spec,
                 kernel_sizes,
                 hidden_channels,
                 strides,
                 hidden_sizes=(32, 32),
                 cnn_hidden_nonlinearity=torch.nn.ReLU,
                 mlp_hidden_nonlinearity=torch.nn.ReLU,
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
                 name='DiscreteCNNPolicy'):

        super().__init__(env_spec, name)
        self._env_spec = env_spec
        self._input_shape = env_spec.observation_space.shape
        self._output_dim = env_spec.action_space.flat_dim
        self._is_image = isinstance(self._env_spec.observation_space,
                                    akro.Image)

        self._cnn_module = DiscreteCNNModule(
            self._input_shape, self._output_dim, kernel_sizes, hidden_channels,
            strides, hidden_sizes, cnn_hidden_nonlinearity,
            mlp_hidden_nonlinearity, hidden_w_init, hidden_b_init, paddings,
            padding_mode, max_pool, pool_shape, pool_stride,
            output_nonlinearity, output_w_init, output_b_init,
            layer_normalization, self._is_image)

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations(torch.Tensor): Batch of observations of shape
                :math:`(N, O)`. Observations should be flattened even
                if they are images as the underlying Q network handles
                unflattening.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors.
                Do not need to be detached, and can be on any device.
        """
        observations = self._env_spec.observation_space.unflatten_n(
            observations)
        if isinstance(self._env_spec.observation_space, akro.Image):
            observations = torch.div(observations, 255.0)

        observations = torch.Tensor(observations[0])
        output = self._cnn_module(observations)
        logits = torch.softmax(output, axis=1)
        dist = torch.distributions.Bernoulli(logits=logits)
        return dist, {}
