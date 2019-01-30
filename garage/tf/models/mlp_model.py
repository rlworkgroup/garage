"""MLP model in TensorFlow using tf.keras.models.Model."""
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from garage.tf.models import PickableModel


class MLPModel(PickableModel):
    """
    MLP model.

    Args:
        input_var: Input tf.Tensor to the MLP.
        output_dim: Dimension of the network output.
        hidden_sizes: Output dimension of dense layer(s).
        scope: Name scope of the mlp.
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        hidden_w_init: Initializer function for the weight
                    of intermediate dense layer(s).
        hidden_b_init: Initializer function for the bias
                    of intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
        output_w_init: Initializer function for the weight
                    of output dense layer(s).
        output_b_init: Initializer function for the bias
                    of output dense layer(s).
        batch_normalization: Bool for using batch normalization or not.

    Return:
        The MLP Model.
    """

    def __init__(self,
                 input_var,
                 output_dim,
                 hidden_sizes,
                 scope="mlp",
                 hidden_nonlinearity="relu",
                 hidden_w_init="glorot_uniform",
                 hidden_b_init="zeros",
                 output_nonlinearity=None,
                 output_w_init="glorot_uniform",
                 output_b_init="zero",
                 batch_normalization=False):

        self._output_dim = output_dim
        self._hidden_sizes = hidden_sizes
        self._scope = scope
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._batch_normalization = batch_normalization

        self.model = self.build_model(input_var)

    def build_model(self, input_var):
        """Build model."""
        _out = input_var
        for idx, hidden_size in enumerate(self._hidden_sizes):
            _out = Dense(
                units=hidden_size,
                activation=self._hidden_nonlinearity,
                kernel_initializer=self._hidden_w_init,
                bias_initializer=self._hidden_b_init,
                name=self._scope + "/hidden_{}".format(idx))(_out)
            if self._batch_normalization:
                _out = BatchNormalization(
                    hidden_size, activation=self._hidden_nonlinearity)(_out)
        _out = Dense(
            units=self._output_dim,
            activation=self._output_nonlinearity,
            kernel_initializer=self._output_w_init,
            bias_initializer=self._output_b_init,
            name=self._scope + "/output")(_out)

        return Model(inputs=input_var, outputs=_out)
