from rllab.algos import TRPO
from rllab.baselines import LinearFeatureBaseline
from rllab.envs.box2d import CartpoleEnv
from rllab.envs import normalize
from rllab.policies import GaussianGRUPolicy
from rllab.optimizers import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.misc import run_experiment_lite


def run_task(*_):
    env = normalize(CartpoleEnv())

    policy = GaussianGRUPolicy(
        env_spec=env.spec,
    )

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
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
    )
    algo.train()


run_experiment_lite(
    run_task,
    n_parallel=1,
    seed=1,
)
