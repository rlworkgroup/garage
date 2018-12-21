"""
This is an example to train a task with VPG algorithm.

Here it runs CartpoleEnv with 100 iterations.

Results:
    AverageReturn: 1000
    RiseTime: itr 60
"""

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.experiment import run_experiment
from garage.tf.algos import VPG
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy


def run_task(*_):
    """Wrap VPG training task in the run_task function."""
    env = TfEnv(normalize(CartpoleEnv()))

    policy = GaussianMLPPolicy(
        name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=10000,
        max_path_length=100,
        n_itr=100,
        discount=0.99,
        optimizer_args=dict(tf_optimizer_args=dict(learning_rate=0.01, )))
    algo.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)
