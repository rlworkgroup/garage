import numpy as np

from garage.algos import TRPO
from garage.baselines import LinearFeatureBaseline
from garage.envs.mujoco.sawyer import BinSortingEnv
from garage.envs.mujoco.sawyer import BlockStackingEnv
from garage.envs.mujoco.sawyer import PickAndPlaceEnv
<<<<<<< HEAD
=======
from garage.envs.mujoco.sawyer import ReachEnv
>>>>>>> add reacher
from garage.envs.util import spec
from garage.misc.instrument import run_experiment
from garage.policies import GaussianMLPPolicy


def run_bin_sorting(*_):
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


def test_env():
    initial_goal = np.array([0.6, -0.1, 0.80])
    env = ReachEnv(initial_goal)
    for i in range(10):
        env.render()
        action = env.action_space.sample()
        env.step(action)
    env.reset()
    env.close()


test_env()
run_experiment(
    run_bin_sorting,
    n_parallel=2,
    plot=True,
)
