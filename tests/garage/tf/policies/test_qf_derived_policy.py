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
