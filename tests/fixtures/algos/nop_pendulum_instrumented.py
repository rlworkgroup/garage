import gym

from garage.envs import normalize
from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from tests.fixtures.algos.instrumented_nop import InstrumentedNOP
from tests.fixtures.policies import DummyPolicy


def run_task(*_):
    env = normalize(gym.make('Pendulum-v0'))

    policy = DummyPolicy(env_spec=env)

    baseline = LinearFeatureBaseline(env_spec=env)
    algo = InstrumentedNOP(
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
    env.close()


run_experiment(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=6,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode='last',
    # Specifies the seed for the experiment. If this is not provided, a random
    # seed will be used
    seed=1,
    plot=True,
)
