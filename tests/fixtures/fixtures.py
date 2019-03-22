import gc
import unittest

import tensorflow as tf

from garage.experiment import deterministic
from garage.logger import logger
from tests.fixtures.logger import NullOutput


class TfTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.sess.__enter__()

    def tearDown(self):
        if tf.get_default_session() is not self.sess:
            self.sess.close()
        else:
            self.sess.close()
            self.sess.__exit__(None, None, None)
        del self.sess
        gc.collect()


class TfGraphTestCase(unittest.TestCase):
    def setUp(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.sess.__enter__()
        logger.add_output(NullOutput())
        deterministic.set_seed(1)

    def tearDown(self):
        logger.remove_all()
        if tf.get_default_session() is not self.sess:
            self.sess.close()
        else:
            self.sess.close()
            self.sess.__exit__(None, None, None)
        # These del are crucial to prevent ENOMEM in the CI
        # b/c TensorFlow does not release memory explicitly
        del self.graph
        del self.sess
        gc.collect()
