import pickle
from unittest import mock

import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.models import Model
from garage.tf.policies import DeterministicMLPPolicyWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class SimpleMLPModel(Model):
    """Simple MLPModel for testing."""

    def __init__(self, name, output_dim, *args, **kwargs):
        super().__init__(name)
        self.output_dim = output_dim

    def _build(self, obs_input):
        return tf.constant(
            1., shape=(
                1,
                self.output_dim,
            ))


class TestDeterministicMLPPolicyWithModel(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.env = TfEnv(DummyBoxEnv())
        self.env.reset()
        self.obs, _, _, _ = self.env.step(1)
        with mock.patch(
                'garage.tf.policies.deterministic_mlp_policy_with_model.MLPModel',  # noqa: E501
                new=SimpleMLPModel):
            self.policy = DeterministicMLPPolicyWithModel(
                env_spec=self.env.spec)

    def test_get_action(self):
        action, _ = self.policy.get_action(self.obs)
        assert self.env.action_space.contains(np.asarray([action]))

        out = self.sess.run(
            self.policy.model.networks['default'].output,
            feed_dict={
                self.policy.model.networks['default'].input: [self.obs]
            })
        assert action == out

        actions, _ = self.policy.get_actions([self.obs])
        assert self.env.action_space.contains(actions)

    def test_get_action_sym(self):
        obs_dim = self.env.spec.observation_space.flat_dim
        state_input = tf.placeholder(tf.float32, shape=(None, obs_dim))
        action_sym = self.policy.get_action_sym(state_input, name="action_sym")

        action1, _ = self.policy.get_action(self.obs)
        action2 = self.sess.run(
            action_sym, feed_dict={state_input: [self.obs]})
        assert action1 == action2

    def test_is_pickleable(self):
        p = pickle.dumps(self.policy)
        action1, _ = self.policy.get_action(self.obs)

        with tf.Session(graph=tf.Graph()):
            policy_pickled = pickle.loads(p)
            action2, _ = policy_pickled.get_action(self.obs)
            assert self.env.action_space.contains(np.asarray([action2]))
            assert action1 == action2
