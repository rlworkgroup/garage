from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.experiment import run_experiment
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy
from tests.fixtures.theano.instrumented_trpo import InstrumentedTRPO


def run_task(*_):
    env = TheanoEnv(normalize(CartpoleEnv()))

    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = InstrumentedTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=4,
        discount=0.99,
        step_size=0.01,
        plot=True)
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
