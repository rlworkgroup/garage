"""This script creates a unittest that tests qf-derived policy."""
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

    def test_deterministic_mlp_policy_with_model_get_action(self):
        obs, _, _, _ = self.env.step(1)
        action, _ = self.policy.get_action(obs)
        assert self.env.action_space.contains(np.asarray([action]))
        actions, _ = self.policy.get_actions([obs])
        for action in actions:
            assert self.env.action_space.contains(np.asarray([action]))

    def test_deterministic_mlp_policy_with_model_is_pickleable(self):
        with tf.Session(graph=tf.Graph()):
            p = pickle.dumps(self.policy)
            policy_pickled = pickle.loads(p)
            obs, _, _, _ = self.env.step(1)
            action, _ = policy_pickled.get_action(obs)
            assert self.env.action_space.contains(np.asarray([action]))
