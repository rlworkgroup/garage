from garage.algos import TRPO
from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.envs.util import spec
from garage.misc.instrument import run_experiment_lite
from garage.optimizers import ConjugateGradientOptimizer
from garage.optimizers import FiniteDifferenceHvp
from garage.policies import GaussianGRUPolicy


def run_task(*_):
    env = normalize(CartpoleEnv())

    policy = GaussianGRUPolicy(env_spec=spec(env), )

    baseline = LinearFeatureBaseline(env_spec=spec(env))

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=10,
        discount=0.99,
        step_size=0.01,
        optimizer=ConjugateGradientOptimizer(
            hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)))
    algo.train()


run_experiment_lite(
    run_task,
    n_parallel=1,
    seed=1,
)
