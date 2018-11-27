from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.experiment import run_experiment
from garage.theano.algos import TRPO
from garage.theano.envs import TheanoEnv
from garage.theano.optimizers import ConjugateGradientOptimizer
from garage.theano.optimizers import FiniteDifferenceHvp
from garage.theano.policies import GaussianGRUPolicy


def run_task(*_):
    env = TheanoEnv(normalize(CartpoleEnv()))

    policy = GaussianGRUPolicy(env_spec=env.spec, )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

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


run_experiment(
    run_task,
    n_parallel=1,
    seed=1,
)
