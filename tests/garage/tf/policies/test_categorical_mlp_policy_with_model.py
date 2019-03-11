import pickle

import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicyWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class TestCategoricalMLPPolicyWithModel(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.env = TfEnv(DummyDiscreteEnv())
        self.policy = CategoricalMLPPolicyWithModel(env_spec=self.env.spec)
        self.env.reset()

    def test_get_action(self):
        obs, _, _, _ = self.env.step(1)
        action, _ = self.policy.get_action(obs)
        assert self.env.action_space.contains(action)
        actions, _ = self.policy.get_actions([obs])
        for action in actions:
            assert self.env.action_space.contains(action)

    def test_is_pickleable(self):
        obs, _, _, _ = self.env.step(1)
        action1, _ = self.policy.get_action(obs)
        p = pickle.dumps(self.policy)

        with tf.Session(graph=tf.Graph()):
            policy_pickled = pickle.loads(p)
            action2, _ = policy_pickled.get_action(obs)
            assert self.env.action_space.contains(action2)

            prob1 = self.policy.dist_info([obs])
            prob2 = policy_pickled.dist_info([obs])
            assert np.array_equal(prob1['prob'], prob2['prob'])

    def test_dist_info_sym(self):
        obs, _, _, _ = self.env.step(1)

        obs_dim = self.env.spec.observation_space.flat_dim
        state_input = tf.placeholder(tf.float32, shape=(None, obs_dim))
        dist1 = self.policy.dist_info_sym(state_input, name="policy2")

        action1, prob1 = self.policy.get_action(obs)
        prob1_dist_info = self.policy.dist_info([obs])

        prob2 = self.sess.run(dist1['prob'], feed_dict={state_input: [obs]})

        assert np.array_equal(prob2, prob1['prob'])
        assert np.array_equal(prob1_dist_info['prob'], prob1['prob'])
