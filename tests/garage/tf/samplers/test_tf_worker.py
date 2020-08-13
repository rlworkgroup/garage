import tensorflow as tf

from garage.envs import GymEnv
from garage.experiment import LocalTFRunner
from garage.sampler import DefaultWorker
from garage.tf.samplers import TFWorkerWrapper

from tests.fixtures import snapshot_config
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestTFWorker:

    def test_tf_worker_with_default_session(self):
        with LocalTFRunner(snapshot_config):
            tf_worker = TFWorkerWrapper()
            worker = DefaultWorker(seed=1,
                                   max_episode_length=100,
                                   worker_number=1)
            worker.update_env(GymEnv(DummyBoxEnv()))
            tf_worker._inner_worker = worker
            tf_worker.worker_init()
            assert tf_worker._sess == tf.compat.v1.get_default_session()
        assert tf_worker._sess._closed

    def test_tf_worker_without_default_session(self):
        tf_worker = TFWorkerWrapper()
        worker = DefaultWorker(seed=1, max_episode_length=100, worker_number=1)
        worker.update_env(GymEnv(DummyBoxEnv()))
        tf_worker._inner_worker = worker
        tf_worker.worker_init()
        assert tf_worker._sess == tf.compat.v1.get_default_session()
        tf_worker.shutdown()
        assert tf_worker._sess._closed
