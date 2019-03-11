import pickle

import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import DeterministicMLPPolicyWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestDeterministicMLPPolicyWithModel(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.env = TfEnv(DummyBoxEnv())
        self.policy = DeterministicMLPPolicyWithModel(env_spec=self.env.spec)
        self.env.reset()

    def test_get_action(self):
        obs, _, _, _ = self.env.step(1)
        action, _ = self.policy.get_action(obs)
        assert self.env.action_space.contains(np.asarray([action]))
        actions, _ = self.policy.get_actions([obs])
        assert self.env.action_space.contains(actions)

    def test_get_action_sym(self):
        obs, _, _, _ = self.env.step(1)

        obs_dim = self.env.spec.observation_space.flat_dim
        state_input = tf.placeholder(tf.float32, shape=(None, obs_dim))
        action_sym = self.policy.get_action_sym(state_input, name="action_sym")

        action1, _ = self.policy.get_action(obs)
        action2 = self.sess.run(action_sym, feed_dict={state_input: [obs]})
        assert action1 == action2

    def test_is_pickleable(self):
        p = pickle.dumps(self.policy)
        obs, _, _, _ = self.env.step(1)
        action1, _ = self.policy.get_action(obs)

        with tf.Session(graph=tf.Graph()):
            policy_pickled = pickle.loads(p)
            action2, _ = policy_pickled.get_action(obs)
            assert self.env.action_space.contains(np.asarray([action2]))
            assert action1 == action2
