"""Testing for sawyer envrionments. """
import collections
from copy import copy
import unittest

import numpy as np

from garage.baselines import LinearFeatureBaseline
from garage.envs.mujoco.sawyer import BinSortingEnv
from garage.envs.mujoco.sawyer import BlockStackingEnv
from garage.envs.mujoco.sawyer import PickAndPlaceEnv
from garage.envs.mujoco.sawyer import ReacherEnv
from garage.theano.algos import TRPO
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy
from tests.helpers import step_env


def run_bin_sorting(*_):
    """Run TRPO for bin sorting env. """

    env = TheanoEnv(BinSortingEnv())

    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        batch_size=4000,
        max_path_length=2000,
        baseline=baseline,
        n_itr=1000,
        discount=0.99,
        step_size=0.01,
        plot=True,
        force_batch_sampler=True,
    )
    algo.train()


def run_block_stacking(*_):
    """Run TRPO with block stacking. """
    env = TheanoEnv(BlockStackingEnv())

    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        batch_size=4000,
        max_path_length=2000,
        baseline=baseline,
        n_itr=1000,
        discount=0.99,
        step_size=0.01,
        plot=True,
        force_batch_sampler=True,
    )
    algo.train()


def run_pick_and_place(*_):
    initial_goal = np.array([0.6, -0.1, 0.80])
    env = TheanoEnv(PickAndPlaceEnv(initial_goal))
    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        batch_size=4000,
        max_path_length=2000,
        baseline=baseline,
        n_itr=1000,
        discount=0.99,
        step_size=0.01,
        plot=True,
        force_batch_sampler=True,
    )
    algo.train()


class TestSawyerEnvs(unittest.TestCase):
    def test_reacher(self):
        """Testing for reacher."""
        tasks = [(0.3, -0.3, 0.30), (0.3, 0.3, 0.30)]

        env = ReacherEnv(goal_position=tasks[0])
        step_env(env, n=5, render=True)

    def test_pnp(self):
        """Testing for pick and place."""

        env = PickAndPlaceEnv()
        step_env(env, n=5, render=True)

    def test_does_not_modify_action(self):
        tasks = [(0.3, -0.3, 0.30), (0.3, 0.3, 0.30)]
        envs = [ReacherEnv(goal_position=tasks[0]), PickAndPlaceEnv()]
        for env in envs:
            env.reset()
            a = env.action_space.sample()
            a_copy = copy(a)
            env.step(a)
            if isinstance(a, collections.Iterable):
                self.assertEquals(a.all(), a_copy.all())
            else:
                self.assertEquals(a, a_copy)
        env.close()
