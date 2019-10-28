import gc
import os

from dowel import logger
import tensorflow as tf

from garage.experiment import deterministic
from garage.experiment.snapshotter import SnapshotConfig
from tests.fixtures.logger import NullOutput

path = os.path.join(os.getcwd(), 'data/local/experiment')
snapshot_config = SnapshotConfig(snapshot_dir=path,
                                 snapshot_mode='last',
                                 snapshot_gap=1)


class TfTestCase:

    def setup_method(self):
        self.sess = tf.compat.v1.Session()
        self.sess.__enter__()

    def teardown_method(self):
        self.sess.__exit__(None, None, None)
        self.sess.close()
        del self.sess
        gc.collect()


class TfGraphTestCase:

    def setup_method(self):
        self.graph = tf.Graph()
        for c in self.graph.collections:
            self.graph.clear_collection(c)
        self.graph_manager = self.graph.as_default()
        self.graph_manager.__enter__()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.sess_manager = self.sess.as_default()
        self.sess_manager.__enter__()
        self.sess.__enter__()
        logger.add_output(NullOutput())
        deterministic.set_seed(1)

        # initialize global singleton_pool for each test case
        from garage.sampler import singleton_pool
        singleton_pool.initialize(1)

    def teardown_method(self):
        logger.remove_all()
        self.sess.__exit__(None, None, None)
        self.sess_manager.__exit__(None, None, None)
        self.graph_manager.__exit__(None, None, None)
        self.sess.close()

        # These del are crucial to prevent ENOMEM in the CI
        # b/c TensorFlow does not release memory explicitly
        del self.graph
        del self.sess
        gc.collect()
