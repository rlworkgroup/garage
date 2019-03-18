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
        model_path = 'garage.tf.policies.' + \
                     'categorical_mlp_policy_with_model.MLPModel'
        with mock.patch(model_path, new=SimpleMLPModel):
            self.policy = CategoricalMLPPolicyWithModel(env_spec=self.env.spec)

    @mock.patch('numpy.random.rand')
    def test_get_action(self, mock_rand):
        mock_rand.return_value = 1
        action, _ = self.policy.get_action(self.obs)
        assert self.env.action_space.contains(action)
        assert action == 0

        actions, _ = self.policy.get_actions([self.obs])
        for action in actions:
            assert self.env.action_space.contains(action)
            assert action == 0

    def test_dist_info(self):
        policy_prob = self.policy.dist_info([self.obs])
        assert np.array_equal(policy_prob['prob'], [1., 1.])

    def test_dist_info_sym(self):
        obs_dim = self.env.spec.observation_space.flat_dim
        state_input = tf.placeholder(tf.float32, shape=(None, obs_dim))
        dist1 = self.policy.dist_info_sym(state_input, name="policy2")

        prob = self.sess.run(
            dist1['prob'], feed_dict={state_input: [self.obs]})

        assert np.array_equal(prob, [1., 1.])

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
