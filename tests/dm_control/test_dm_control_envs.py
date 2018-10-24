import collections
from copy import copy
import pickle
import unittest

from dm_control.suite import ALL_TASKS

from tests.fixtures import DmParameterizedTestCase
from tests.helpers import step_env


class DmTestCase(DmParameterizedTestCase):
    def test_can_step_and_render(self):
        ob_space = self.env.observation_space
        act_space = self.env.action_space
        ob = self.env.reset()
        assert ob_space.contains(ob)
        a = act_space.sample()
        assert act_space.contains(a)
        step_env(self.env, n=10, render=True)

    def test_pickling(self):
        round_trip = pickle.loads(pickle.dumps(self.env))
        assert round_trip
        step_env(round_trip)

    def test_all_does_not_modify_actions(self):
        a = self.env.action_space.sample()
        a_copy = copy(a)
        self.env.step(a)
        if isinstance(a, collections.Iterable):
            self.assertEquals(a.all(), a_copy.all())
        else:
            self.assertEquals(a, a_copy)


for task in ALL_TASKS:
    suite = unittest.TestSuite()
    suite.addTest(DmParameterizedTestCase.parameterize(DmTestCase, param=task))
    unittest.TextTestRunner(verbosity=2).run(suite)
