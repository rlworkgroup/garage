import unittest

import tensorflow as tf

import garage.misc.logger as logger
from garage.misc.tensorboard_output import TensorBoardOutput


class GarageTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session(graph=tf.Graph())
        self.sess.__enter__()
        logger._tensorboard = TensorBoardOutput()

    def tearDown(self):
        self.sess.close()
