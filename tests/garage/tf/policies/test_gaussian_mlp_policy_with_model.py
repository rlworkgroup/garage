import pickle

import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicyWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestGaussianMLPPolicyWithModel(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.box_env = TfEnv(DummyBoxEnv())
        self.policy = GaussianMLPPolicyWithModel(
            env_spec=self.box_env, init_std=1.1, name="P1")

        self.obs = [self.box_env.reset()]
        self.obs_ph = tf.placeholder(tf.float32, shape=(None, 1))

        self.dist1_sym = self.policy.dist_info_sym(self.obs_ph, name='p1_sym')

    def test_get_action(self):
        action, _ = self.policy.get_action(self.obs)
        assert self.box_env.action_space.contains(np.array(action[0]))

        actions, _ = self.policy.get_actions(self.obs)
        assert self.box_env.action_space.contains(np.array(actions[0]))

    def test_dist_info_sym(self):
        _, dist1 = self.policy.get_action(self.obs)
        dist2 = self.sess.run(
            self.dist1_sym, feed_dict={self.obs_ph: self.obs})

        assert np.array_equal(dist1, dist2)

    def test_is_pickleable(self):
        with tf.Session(graph=tf.Graph()) as sess:
            policy = GaussianMLPPolicyWithModel(env_spec=self.box_env)
            # model is built in GaussianMLPPolicyWithModel.__init__
            outputs = sess.run(
                policy.model.networks['default'].sample,
                feed_dict={policy.model.networks['default'].input: self.obs})
            p = pickle.dumps(policy)

        with tf.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            outputs2 = sess.run(
                policy_pickled.model.networks['default'].sample,
                feed_dict={
                    policy_pickled.model.networks['default'].input: self.obs
                })

        assert np.array_equal(outputs, outputs2)
