import pickle

import pytest
import tensorflow as tf

from garage.envs import GymEnv
from garage.envs.wrappers import AtariEnv
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteCNNQFunction

# yapf: disable
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import (DummyDictEnv,
                                       DummyDiscreteEnv,
                                       DummyDiscretePixelEnvBaselines)
from tests.fixtures.q_functions import SimpleQFunction

# yapf: enable


class TestQfDerivedPolicy(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.env = GymEnv(DummyDiscreteEnv())
        self.qf = SimpleQFunction(self.env.spec)
        self.policy = DiscreteQfDerivedPolicy(env_spec=self.env.spec,
                                              qf=self.qf)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.env.reset()

    def test_discrete_qf_derived_policy(self):
        obs = self.env.step(1).observation
        action, _ = self.policy.get_action(obs)
        assert self.env.action_space.contains(action)
        actions, _ = self.policy.get_actions([obs])
        for action in actions:
            assert self.env.action_space.contains(action)

    def test_get_param(self):
        with tf.compat.v1.variable_scope('SimpleQFunction', reuse=True):
            return_var = tf.compat.v1.get_variable('return_var')
        assert self.policy.get_param_values() == return_var.eval()

    def test_is_pickleable(self):
        with tf.compat.v1.variable_scope('SimpleQFunction', reuse=True):
            return_var = tf.compat.v1.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())
        obs = self.env.step(1).observation
        action1, _ = self.policy.get_action(obs)

        p = pickle.dumps(self.policy)
        with tf.compat.v1.Session(graph=tf.Graph()):
            policy_pickled = pickle.loads(p)
            action2, _ = policy_pickled.get_action(obs)
            assert action1 == action2

    def test_does_not_support_dict_obs_space(self):
        """Test that policy raises error if passed a dict obs space."""
        env = GymEnv(DummyDictEnv(act_space_type='discrete'))
        with pytest.raises(ValueError):
            qf = SimpleQFunction(env.spec,
                                 name='does_not_support_dict_obs_space')
            DiscreteQfDerivedPolicy(env_spec=env.spec, qf=qf)

    def test_invalid_action_spaces(self):
        """Test that policy raises error if passed a dict obs space."""
        env = GymEnv(DummyDictEnv(act_space_type='box'))
        with pytest.raises(ValueError):
            qf = SimpleQFunction(env.spec)
            DiscreteQfDerivedPolicy(env_spec=env.spec, qf=qf)


class TestQfDerivedPolicyImageObs(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.env = GymEnv(AtariEnv(DummyDiscretePixelEnvBaselines()),
                          is_image=True)
        self.qf = DiscreteCNNQFunction(env_spec=self.env.spec,
                                       filters=((1, (1, 1)), ),
                                       strides=(1, ),
                                       dueling=False)
        self.policy = DiscreteQfDerivedPolicy(env_spec=self.env.spec,
                                              qf=self.qf)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.env.reset()

    def test_obs_unflattened(self):
        """Test if a flattened image obs is passed to get_action
           then it is unflattened.
        """
        obs = self.env.observation_space.sample()
        action, _ = self.policy.get_action(
            self.env.observation_space.flatten(obs))
        self.env.step(action)
