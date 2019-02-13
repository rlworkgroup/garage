import pickle
import unittest

import numpy as np
import tensorflow as tf

from garage.tf.models.gaussian_mlp_model import GaussianMLPModel


class TestNewModelPickling(unittest.TestCase):
    def test_model_pickle(self):
        sess = tf.Session()
        with sess:
            input_var = tf.placeholder(tf.float32, shape=(None, 5))
            model = GaussianMLPModel(output_dim=2)
            model.build(input_var)
            data = np.random.random((3, 5))
            results = sess.run(model.outputs[:2], feed_dict={input_var: data})
            model_data = pickle.dumps(model)

        sess.close()
        tf.reset_default_graph()

        new_sess = tf.Session()
        with new_sess as sess:
            input_var = tf.placeholder(tf.float32, shape=(None, 5))
            model_pickled = pickle.loads(model_data)
            model_pickled.build(input_var)

            results2 = sess.run(
                model_pickled.outputs[:2], feed_dict={input_var: data})

            # result[0] is sample(), which is supposed to be different
            assert np.array_equal(results[1:], results2[1:])
            assert isinstance(model.dist,
                              tf.contrib.distributions.MultivariateNormalDiag)

    def test_model_pickle_same_parameters(self):
        g = GaussianMLPModel(output_dim=2, name="some_model")

        sess = tf.Session()
        with sess.as_default():
            state = tf.placeholder(shape=[None, 10, 5], dtype=tf.float32)
            g.build(state)

            g.parameters = {
                k: np.zeros_like(v)
                for k, v in g.parameters.items()
            }
            all_one = {k: np.ones_like(v) for k, v in g.parameters.items()}
            g.parameters = all_one
            h_data = pickle.dumps(g)

        sess.close()
        tf.reset_default_graph()

        sess = tf.Session()
        with sess.as_default():
            hh = pickle.loads(h_data)
            state = tf.placeholder(shape=[None, 10, 5], dtype=tf.float32)
            hh.build(state)

            np.testing.assert_equal(all_one, hh.parameters)
