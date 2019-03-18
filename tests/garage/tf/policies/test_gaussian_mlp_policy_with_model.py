import pickle
from unittest import mock

import numpy as np
import tensorflow as tf

from garage.tf.distributions import DiagonalGaussian
from garage.tf.envs import TfEnv
from garage.tf.models import Model
from garage.tf.policies import GaussianMLPPolicyWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class SimpleGaussianMLPModel(Model):
    """Simple GaussianMLPModel for testing."""

    def __init__(self, name, output_dim, *args, **kwargs):
        super().__init__(name)
        self.output_dim = output_dim

    def network_output_spec(self):
        return ['sample', 'mean', 'log_std', 'dist']

    def _build(self, obs_input):
        mean = tf.constant(0.5, shape=((1, self.output_dim)))
        log_std = tf.constant(0.5, shape=((1, self.output_dim)))
        action = mean + log_std * 0.5
        dist = DiagonalGaussian(self.output_dim)
        # action will be 0.5 + 0.5 * 0.5 = 0.75
        return action, mean, log_std, dist


class TestGaussianMLPPolicyWithModel(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.env = TfEnv(DummyBoxEnv())
        self.env.reset()
        self.obs, _, _, _ = self.env.step(1)
        self.obs_ph = tf.placeholder(tf.float32, shape=(None, 1))

        model_path = 'garage.tf.policies.' + \
                     'gaussian_mlp_policy_with_model.GaussianMLPModel'
        with mock.patch(model_path, new=SimpleGaussianMLPModel):
            self.policy = GaussianMLPPolicyWithModel(env_spec=self.env.spec)

    def test_get_action(self):
        action, prob = self.policy.get_action(self.obs)
        assert self.env.action_space.contains(np.array(action[0]))
        assert action == 0.75
        assert prob['mean'] == 0.5 and prob['log_std'] == 0.5

        actions, probs = self.policy.get_actions([self.obs])
        assert self.env.action_space.contains(np.array(actions[0]))
        assert actions == 0.75
        assert probs['mean'] == 0.5 and probs['log_std'] == 0.5

    def test_dist_info_sym(self):
        dist1_sym = self.policy.dist_info_sym(self.obs_ph, name='p1_sym')
        prob = self.sess.run(dist1_sym, feed_dict={self.obs_ph: [self.obs]})
        assert prob['mean'] == 0.5 and prob['log_std'] == 0.5

    def test_is_pickleable(self):
        outputs = self.sess.run(
            self.policy.model.networks['default'].sample,
            feed_dict={
                self.policy.model.networks['default'].input: [self.obs]
            })
        p = pickle.dumps(self.policy)
        with tf.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            outputs2 = sess.run(
                policy_pickled.model.networks['default'].sample,
                feed_dict={
                    policy_pickled.model.networks['default'].input: [self.obs]
                })

        assert np.array_equal(outputs, outputs2)
