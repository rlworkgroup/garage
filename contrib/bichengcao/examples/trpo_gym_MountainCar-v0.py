# This doesn't work. After 150 iterations still didn't learn anything.

from rllab.algos import TRPO
from rllab.baselines import LinearFeatureBaseline
from rllab.envs import GymEnv
from rllab.envs import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies import CategoricalMLPPolicy


def run_task(*_):
    env = normalize(GymEnv("MountainCar-v0"))

    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=env.horizon,
        n_itr=150,
        discount=0.99,
        step_size=0.1,
        plot=True,
    )
    algo.train()


run_experiment_lite(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    plot=True,
)
