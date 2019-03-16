import pickle
from unittest import mock

import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.models import Model
from garage.tf.policies import CategoricalMLPPolicyWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv


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


class TestCategoricalMLPPolicyWithModel(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.env = TfEnv(DummyDiscreteEnv())
        self.env.reset()
        self.obs, _, _, _ = self.env.step(1)
        with mock.patch(
                'garage.tf.policies.categorical_mlp_policy_with_model.MLPModel',  # noqa: E501
                new=SimpleMLPModel):
            self.policy = CategoricalMLPPolicyWithModel(env_spec=self.env.spec)

    def test_get_action(self):
        action, _ = self.policy.get_action(self.obs)
        assert self.env.action_space.contains(action)

        actions, _ = self.policy.get_actions([self.obs])
        for action in actions:
            assert self.env.action_space.contains(action)

    def test_dist_info(self):
        policy_prob = self.policy.dist_info([self.obs])

        empirical_prob = self.sess.run(
            self.policy.model.networks['default'].output,
            feed_dict={
                self.policy.model.networks['default'].input: [self.obs]
            })

        assert np.array_equal(empirical_prob, policy_prob['prob'])

    def test_dist_info_sym(self):
        obs_dim = self.env.spec.observation_space.flat_dim
        state_input = tf.placeholder(tf.float32, shape=(None, obs_dim))
        dist1 = self.policy.dist_info_sym(state_input, name="policy2")

        action1, prob1 = self.policy.get_action(self.obs)
        prob1_dist_info = self.policy.dist_info([self.obs])

        prob2 = self.sess.run(
            dist1['prob'], feed_dict={state_input: [self.obs]})

        assert np.array_equal(prob2, prob1['prob'])
        assert np.array_equal(prob1_dist_info['prob'], prob1['prob'])

    def test_is_pickleable(self):
        action1, _ = self.policy.get_action(self.obs)
        p = pickle.dumps(self.policy)

        with tf.Session(graph=tf.Graph()):
            policy_pickled = pickle.loads(p)
            action2, _ = policy_pickled.get_action(self.obs)
            assert self.env.action_space.contains(action2)

            prob1 = self.policy.dist_info([self.obs])
            prob2 = policy_pickled.dist_info([self.obs])
            assert np.array_equal(prob1['prob'], prob2['prob'])
