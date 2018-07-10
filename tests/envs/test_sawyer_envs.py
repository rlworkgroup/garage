"""Testing for sawyer envrionments. """

import numpy as np

from garage.algos import TRPO
from garage.baselines import LinearFeatureBaseline
from garage.envs.mujoco.sawyer import BinSortingEnv
from garage.envs.mujoco.sawyer import BlockStackingEnv
from garage.envs.mujoco.sawyer import PickAndPlaceEnv
from garage.envs.mujoco.sawyer import ReacherEnv
from garage.envs.util import spec
from garage.policies import GaussianMLPPolicy


def run_bin_sorting(*_):
    """Run TRPO for bin sorting env. """

    env = BinSortingEnv()

    policy = GaussianMLPPolicy(env_spec=spec(env), hidden_sizes=(32, 32))
    baseline = LinearFeatureBaseline(env_spec=spec(env))
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
    env = BlockStackingEnv()

    policy = GaussianMLPPolicy(env_spec=spec(env), hidden_sizes=(32, 32))
    baseline = LinearFeatureBaseline(env_spec=spec(env))
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
    env = PickAndPlaceEnv(initial_goal)
    policy = GaussianMLPPolicy(env_spec=spec(env), hidden_sizes=(32, 32))
    baseline = LinearFeatureBaseline(env_spec=spec(env))
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


def test_reacher():
    """Testing for reacher."""

    env = ReacherEnv()
    for i in range(9999):
        env.render()
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
    env.reset()
    env.close()


def test_pnp():
    """Testing for pick and place."""

    env = PickAndPlaceEnv()
    for i in range(9999):
        env.render()
        action = env.action_space.sample()
        env.step(action)
    env.reset()
    env.close()


test_reacher()
test_pnp()
