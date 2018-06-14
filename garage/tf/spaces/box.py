import tensorflow as tf

from garage.spaces import Box as TheanoBox


class Box(TheanoBox):
    def new_tensor_variable(self, name, extra_dims, flatten=True):
        if flatten:
            return tf.placeholder(
                tf.float32,
                shape=[None] * extra_dims + [self.flat_dim],
                name=name)
        return tf.placeholder(
            tf.float32,
            shape=[None] * extra_dims + list(self.shape),
            name=name)

    @property
    def dtype(self):
        return tf.float32
