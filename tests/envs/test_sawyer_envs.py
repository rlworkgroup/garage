"""Testing for sawyer envrionments. """
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
        for i in range(5):
            env.render()
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
        env.reset()
        env.close()

    def test_pnp(self):
        """Testing for pick and place."""

        env = PickAndPlaceEnv()
        for i in range(5):
            env.render()
            action = env.action_space.sample()
            env.step(action)
        env.reset()
        env.close()
