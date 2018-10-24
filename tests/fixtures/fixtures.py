import gc
import unittest

from dm_control.suite import ALL_TASKS
import tensorflow as tf

from garage.envs.dm_control import DmControlEnv
import garage.misc.logger as logger
from garage.misc.tensorboard_output import TensorBoardOutput


class TfTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.sess.__enter__()

    def tearDown(self):
        self.sess.close()
        del self.sess
        gc.collect()


class TfGraphTestCase(unittest.TestCase):
    def setUp(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.sess.__enter__()
        logger._tensorboard = TensorBoardOutput()

    def tearDown(self):
        self.sess.close()
        # These del are crucial to prevent ENOMEM in the CI
        # b/c TensorFlow does not release memory explicitly
        del self.graph
        del self.sess
        gc.collect()


class DmParameterizedTestCase(unittest.TestCase):
    def __init__(self, method_name='runTest', param=ALL_TASKS[0]):
        super().__init__(method_name)
        self.env = DmControlEnv(domain_name=param[0], task_name=param[1])

    @staticmethod
    def parameterize(child_class, param=None):
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(child_class)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(child_class(name, param=param))
        return suite
