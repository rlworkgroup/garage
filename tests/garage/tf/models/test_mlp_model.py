import numpy as np
import tensorflow as tf

from garage.tf.models import MLPModel
from tests.fixtures import TfGraphTestCase


class TestMLPModel(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.input_var = tf.placeholder(tf.float32, shape=(None, 5))
        self.obs = np.ones((1, 5))
        self.output_dim = 1

    def test_output_values(self):
        model = MLPModel(
            output_dim=self.output_dim,
            hidden_sizes=(2, ),
            hidden_nonlinearity=None,
            hidden_w_init=tf.ones_initializer,
            output_w_init=tf.ones_initializer)
        outputs = model.build(self.input_var)
        output = self.sess.run(outputs, feed_dict={self.input_var: self.obs})

        assert output == 10.
