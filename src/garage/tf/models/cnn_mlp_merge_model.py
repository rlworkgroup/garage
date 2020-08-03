"""CNN and MLP Merge Model."""
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.models.cnn_model import CNNModel
from garage.tf.models.cnn_model_max_pooling import CNNModelWithMaxPooling
from garage.tf.models.mlp_merge_model import MLPMergeModel
from garage.tf.models.model import Model


class CNNMLPMergeModel(Model):
    """Convolutional neural network followed by a Multilayer Perceptron.

    Combination of a CNN Model (optionally with max pooling) and an
    MLP Merge model. The CNN accepts the state as an input, while
    the MLP accepts the CNN's output and the action as inputs.

    Args:
        filters (Tuple[Tuple[int, Tuple[int, int]], ...]): Number and dimension
            of filters. For example, ((3, (3, 5)), (32, (3, 3))) means there
            are two convolutional layers. The filter for the first layer have 3
            channels and its shape is (3 x 5), while the filter for the second
            layer have 32 channels and its shape is (3 x 3).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_sizes (tuple[int]): Output dimension of dense layer(s).
            For example, (32, 32) means the MLP of this q-function consists of
            two hidden layers, each with 32 hidden units.
        output_dim (int): Dimension of the network output.
        action_merge_layer (int): The index of layers at which to concatenate
            action inputs with the network. The indexing works like standard
            python list indexing. Index of 0 refers to the input layer
            (observation input) while an index of -1 points to the last
            hidden layer. Default parameter points to second layer from the
            end.
        name (str): Model name, also the variable scope.
        padding (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        max_pooling (bool): Boolean for using max pooling layer or not.
        pool_shapes (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            shape (2, 2).
        pool_strides (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        cnn_hidden_nonlinearity (callable): Activation function for
            intermediate dense layer(s) in the CNN. It should return a
            tf.Tensor. Set it to None to maintain a linear activation.
        cnn_hidden_w_init (callable): Initializer function for the weight of
            intermediate dense layer(s) in the CNN. Function should return a
            tf.Tensor.
        cnn_hidden_b_init (callable): Initializer function for the bias of
            intermediate dense layer(s) in the CNN. Function should return a
            tf.Tensor.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s) in the MLP. It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s) in the MLP. The function should
            return a tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s) in the MLP. The function should
            return a tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer in the MLP. It should return a tf.Tensor. Set it to None
            to maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the MLP. The function should return
            a tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s) in the MLP. The function should return
            a tf.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 filters,
                 strides,
                 hidden_sizes=(256, ),
                 output_dim=1,
                 action_merge_layer=-2,
                 name=None,
                 padding='SAME',
                 max_pooling=False,
                 pool_strides=(2, 2),
                 pool_shapes=(2, 2),
                 cnn_hidden_nonlinearity=tf.nn.relu,
                 cnn_hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 cnn_hidden_b_init=tf.zeros_initializer(),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 output_b_init=tf.zeros_initializer(),
                 layer_normalization=False):
        super().__init__(name)

        if not max_pooling:
            self.cnn_model = CNNModel(
                filters=filters,
                hidden_w_init=cnn_hidden_w_init,
                hidden_b_init=cnn_hidden_b_init,
                strides=strides,
                padding=padding,
                hidden_nonlinearity=cnn_hidden_nonlinearity)
        else:
            self.cnn_model = CNNModelWithMaxPooling(
                filters=filters,
                hidden_w_init=cnn_hidden_w_init,
                hidden_b_init=cnn_hidden_b_init,
                strides=strides,
                padding=padding,
                pool_strides=pool_strides,
                pool_shapes=pool_shapes,
                hidden_nonlinearity=cnn_hidden_nonlinearity)

        self.mlp_merge_model = MLPMergeModel(
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            concat_layer=action_merge_layer,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

    def network_input_spec(self):
        """Network input spec.

        Return:
            list[str]: List of key(str) for the network inputs.

        """
        return ['state', 'action']

    # pylint: disable=arguments-differ
    def _build(self, state, action, name=None):
        """Build the model and return the outputs.

        This builds the model such that the output of the CNN is fed
        to the MLP. The CNN receives the state as the input. The MLP
        receives two inputs, the output of the CNN and the action
        tensor.

        Args:
            state (tf.Tensor): State placeholder tensor of shape
                :math:`(N, O*)`.
            action (tf.Tensor): Action placeholder tensor of shape
                :math:`(N, A*)`.
            name (str): Name of the model.

        Returns:
            tf.Tensor: Output of the model of shape (N, output_dim).

        """
        cnn_out = self.cnn_model.build(state, name=name).outputs
        mlp_out = self.mlp_merge_model.build(cnn_out, action,
                                             name=name).outputs
        return mlp_out
