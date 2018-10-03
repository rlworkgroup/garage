import unittest

from dm_control.suite import ALL_TASKS

from tests.fixtures import DmParameterizedTestCase, DmTestCase

for task in ALL_TASKS:
    suite = unittest.TestSuite()
    suite.addTest(DmParameterizedTestCase.parameterize(DmTestCase, param=task))
    unittest.TextTestRunner(verbosity=2).run(suite)
