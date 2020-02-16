"""An MLP network for encoding context of RL tasks."""

from garage.torch.modules import MLPModule


class MLPEncoder(MLPModule):
    """This MLP network encodes context of RL tasks.

    Context is stored in the terms of observation, action, and reward, and this
    network uses an MLP module for encoding it.

    Args:
        *args: MLPModule arguments.
        **kwargs: MLPModule arguments including:
            input_dim (int) : Dimension of the network input.
            output_dim (int): Dimension of the network output.
            hidden_sizes (list[int]): Output dimension of dense layer(s).
                For example, (32, 32) means this MLP consists of two
                hidden layers, each with 32 hidden units.
            hidden_nonlinearity (callable or torch.nn.Module): Activation
                function for intermediate dense layer(s). It should return a
                torch.Tensor.Set it to None to maintain a linear activation.
            hidden_w_init (callable): Initializer function for the weight
                of intermediate dense layer(s). The function should return a
                torch.Tensor.
            hidden_b_init (callable): Initializer function for the bias
                of intermediate dense layer(s). The function should return a
                torch.Tensor.
            output_nonlinearity (callable or torch.nn.Module): Activation
                function for output dense layer. It should return a
                torch.Tensor. Set it to None to maintain a linear activation.
            output_w_init (callable): Initializer function for the weight
                of output dense layer(s). The function should return a
                torch.Tensor.
            output_b_init (callable): Initializer function for the bias
                of output dense layer(s). The function should return a
                torch.Tensor.
            layer_normalization (bool): Bool for using layer normalization or
                not.

    """

    def reset(self, num_tasks=1):
        """Reset hidden state task size. Not applicable to this encoder.

        Args:
            num_tasks (int): Size of tasks.

        """
