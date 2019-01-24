from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization

# flake8: noqa
# pylint: noqa


def mlp2(input_var,
         output_dim,
         hidden_sizes,
         hidden_nonlinearity='relu',
         hidden_w_init='glorot_uniform',
         hidden_b_init='zeros',
         output_nonlinearity=None,
         output_w_init='glorot_uniform',
         output_b_init='zero',
         batch_normalization=False,
         *args,
         **kwargs):
    x = input_var
    if isinstance(hidden_sizes, int):
        x = Dense(
            units=hidden_sizes,
            activation=hidden_nonlinearity,
            kernel_initializer=hidden_w_init,
            bias_initializer=hidden_b_init)(x)
        if batch_normalization:
            x = BatchNormalization(
                hidden_sizes, activation=hidden_nonlinearity)(x)
    else:
        x = Dense(
            units=hidden_sizes[0],
            activation=hidden_nonlinearity,
            kernel_initializer=hidden_w_init,
            bias_initializer=hidden_b_init)(x)
        if batch_normalization:
            x = BatchNormalization(
                hidden_sizes[0], activation=hidden_nonlinearity)(x)
        for hidden_size in hidden_sizes[1:]:
            x = Dense(
                units=hidden_size,
                activation=hidden_nonlinearity,
                kernel_initializer=hidden_w_init,
                bias_initializer=hidden_b_init)(x)
            if batch_normalization:
                x = BatchNormalization(
                    hidden_size, activation=hidden_nonlinearity)(x)
    x = Dense(
        units=output_dim,
        activation=output_nonlinearity,
        kernel_initializer=output_w_init,
        bias_initializer=output_b_init)(x)
    return Model(inputs=input_var, outputs=x)
