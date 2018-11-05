import sys

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.experiment import run_experiment
from garage.theano.algos import TRPO
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy


def run_task(v):
    env = TheanoEnv(normalize(CartpoleEnv()))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers,
        # each with 32 hidden units.
        hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=40,
        discount=0.99,
        step_size=v["step_size"],
        # Uncomment both lines (this and the plot parameter below) to enable
        # plotting
        plot=True,
    )
    algo.train()


for step_size in [0.01, 0.05, 0.1]:
    for seed in [1, 11, 21, 31, 41]:
        run_experiment(
            run_task,
            exp_prefix="first_exp",
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a
            # random seed will be used
            seed=seed,
            # mode="local",
            mode="ec2",
            variant=dict(step_size=step_size, seed=seed)
            # plot=True,
            # terminate_machine=False,
        )
        sys.exit()
