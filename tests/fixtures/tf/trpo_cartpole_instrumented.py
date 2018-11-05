import tensorflow as tf

from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.experiment import run_experiment
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures.tf.instrumented_trpo import InstrumentedTRPO


def run_task(*_):
    env = TfEnv(normalize(CartpoleEnv()))
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=tf.nn.tanh,
        output_nonlinearity=None,
    )
    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args=dict(hidden_sizes=(32, 32)),
    )
    algo = InstrumentedTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=1024,
        max_path_length=100,
        n_itr=4,
        discount=0.99,
        gae_lambda=0.98,
        policy_ent_coeff=0.0,
        plot=True,
    )
    algo.train()


run_experiment(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=6,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random
    # seed will be used
    seed=1,
    plot=True,
)
