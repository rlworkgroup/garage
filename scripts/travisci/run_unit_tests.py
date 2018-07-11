import os.path as osp
import sys

import nose2

# failed
FAILED_TESTS = [
    'test_spaces', 'test_networks', 'test_baselines', 'test_algos',
    'regression_tests.test_issue_3', 'algos.test_trpo', 'test_stateful_pool'
]

# slow but successfully
SLOW_TESTS = [
    'test_normalized_gym', 'test_dynamics_rand',
    'regression_tests.test_benchmark_ddpg', 'envs.test_envs',
    'envs.test_sawyer_envs', 'test_dm_control'
]

AVAILABLE_TESTS = [
    'test_serializable', 'test_sampler', 'test_instrument',
    'envs.test_maze_env'
]

tests_dir = osp.join(
    osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'tests')

test_run = nose2.discover(
    argv=['-v', '-s', tests_dir] + AVAILABLE_TESTS, exit=False)

passed = False

if test_run.result.wasSuccessful():
    passed = True

sys.exit(not passed)
