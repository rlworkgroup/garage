import unittest

import tensorflow as tf

import garage.misc.logger as logger
from garage.misc.tensorboard_output import TensorBoardOutput


class GarageTestCase(unittest.TestCase):
    def setUp(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.sess.__enter__()
        logger._tensorboard = TensorBoardOutput()

    def tearDown(self):
        self.sess.close()
        # These del are crucial to prevent ENOMEM in the CI
        del self.graph
        del self.sess
