"""Launcher file for trpo sawyer reacher training."""

from garage.baselines import LinearFeatureBaseline
from garage.envs.mujoco.sawyer import ReacherEnv
from garage.misc.instrument import run_experiment
from garage.theano.algos import TRPO
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy


def run(*_):
    """Stub method for running trpo."""
    env = TheanoEnv(
        ReacherEnv(control_method='position_control', sparse_reward=False))
    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))
    baseline = LinearFeatureBaseline(env_spec=env.spec)
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


run_experiment(
    run,
    n_parallel=2,
    plot=True,
)
