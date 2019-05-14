import pickle

import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import DiscreteQfDerivedPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv
from tests.fixtures.q_functions import SimpleQFunction


class TestQfDerivedPolicy(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.env = TfEnv(DummyDiscreteEnv())
        self.qf = SimpleQFunction(self.env.spec)
        self.policy = DiscreteQfDerivedPolicy(
            env_spec=self.env.spec, qf=self.qf)
        self.sess.run(tf.global_variables_initializer())
        self.env.reset()

    def test_discrete_qf_derived_policy(self):
        obs, _, _, _ = self.env.step(1)
        action = self.policy.get_action(obs)
        assert self.env.action_space.contains(action)
        actions = self.policy.get_actions([obs])
        for action in actions:
            assert self.env.action_space.contains(action)

    def test_is_pickleable(self):
        with tf.variable_scope('SimpleQFunction/SimpleMLPModel', reuse=True):
            return_var = tf.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())
        obs, _, _, _ = self.env.step(1)
        action1 = self.policy.get_action(obs)

        p = pickle.dumps(self.policy)
        with tf.Session(graph=tf.Graph()):
            policy_pickled = pickle.loads(p)
            action2 = policy_pickled.get_action(obs)
            assert action1 == action2
