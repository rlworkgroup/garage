"""Launcher file for trpo sawyer reacher training. """

from garage.algos import TRPO
from garage.baselines import LinearFeatureBaseline
from garage.envs.mujoco.sawyer import ReacherEnv
from garage.envs.util import spec
from garage.misc.instrument import run_experiment
from garage.policies import GaussianMLPPolicy


def run(*_):
    """Method for TRPO with reacher environment."""

    env = ReacherEnv(control_method='position_control', sparse_reward=False)
    policy = GaussianMLPPolicy(env_spec=spec(env), hidden_sizes=(32, 32))
    baseline = LinearFeatureBaseline(env_spec=spec(env))
    algo = TRPO(
        env=env,
        policy=policy,
        batch_size=4000,
        max_path_length=100,
        baseline=baseline,
        n_itr=2500,
        discount=0.99,
        step_size=0.01,
        plot=True,
        force_batch_sampler=True,
    )
    algo.train()


# run()

run_experiment(
    run,
    n_parallel=2,
    plot=True,
)

