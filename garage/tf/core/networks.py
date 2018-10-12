import tensorflow as tf
from garage.core import Serializable


class MLPs(Serializable):
    def __init__(
            self,
            output_dim,
            hidden_sizes,
            hidden_nonlinearity,
            output_nonlinearity,
            name=None,
            hidden_w_init=tf.contrib.layers.xavier_initializer(),
            hidden_b_init=tf.zeros_initializer(),
            output_w_init=tf.contrib.layers.xavier_initializer(),
            output_b_init=tf.zeros_initializer(),
            input_var=None,
            input_shape=None,
            batch_normalization=False,
            weight_normalization=False,
    ):

        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):
            if input_var is None:
                l_in = tf.placeholder(
                    tf.float32,
                    shape=(None, ) + input_shape,  # dynamic batch dimension
                    name="input")
            else:
                l_in = input_var
            self._layers = [l_in]
            l_hid = l_in
            for idx, hidden_size in enumerate(hidden_sizes):
                l_hid = tf.layers.dense(
                    inputs=l_hid,
                    units=hidden_size,
                    activation=hidden_nonlinearity,
                    kernel_initializer=hidden_w_init,
                    bias_initializer=hidden_b_init,
                    name="hidden_%d" % idx)

                self._layers.append(l_hid)
            l_out = tf.layers.dense(
                inputs=l_hid,
                units=output_dim,
                activation=output_nonlinearity,
                kernel_initializer=output_w_init,
                bias_initializer=output_b_init,
                name="output")

            self._layers.append(l_out)
            self._l_in = l_in
            self._l_out = l_out

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in

    @property
    def layers(self):
        return self._layers
