import datetime
import os.path as osp
import unittest
import uuid

import dateutil
import joblib
import tensorflow as tf

from garage.baselines import LinearFeatureBaseline
from garage.experiment import LocalRunner
from garage.misc.logger import logger
from garage.sampler.utils import rollout
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy


class TestSnapshot(unittest.TestCase):
    verifyItrs = 3

    @classmethod
    def reset_tf(cls):
        if tf.get_default_session():
            tf.get_default_session().__exit__(None, None, None)
        tf.reset_default_graph()

    @classmethod
    def setUpClass(cls):
        cls.reset_tf()
        # Save snapshot in params.pkl in self.log_dir folder
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        rand_id = str(uuid.uuid4())[:5]
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
        exp_name = 'experiment_%s_%s' % (timestamp, rand_id)
        cls.log_dir = osp.join('/tmp', exp_name)

        cls.prev_log_dir = logger.get_snapshot_dir()
        cls.prev_mode = logger.get_snapshot_mode()

        logger.set_snapshot_dir(cls.log_dir)
        logger.set_snapshot_mode('all')

    @classmethod
    def tearDownClass(cls):
        logger.set_snapshot_dir(cls.prev_log_dir)
        logger.set_snapshot_mode(cls.prev_mode)

    def test_snapshot(self):
        with LocalRunner() as runner:
            logger.reset()
            env = TfEnv(env_name="CartPole-v1")

            policy = CategoricalMLPPolicy(
                name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                max_kl_step=0.01)

            runner.setup(algo, env)
            runner.train(n_epochs=self.verifyItrs, batch_size=4000)

            env.close()

        # Read snapshot from self.log_dir
        # Test the presence and integrity of policy and env
        for i in range(0, self.verifyItrs):
            self.reset_tf()
            with LocalRunner():
                snapshot = joblib.load(osp.join(self.log_dir, f'itr_{i}.pkl'))

                env = snapshot['env']
                policy = snapshot['policy']
                assert env
                assert policy

                rollout(env, policy, animated=False)
