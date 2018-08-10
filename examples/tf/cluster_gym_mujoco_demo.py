import sys

import gym

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.misc.instrument import run_experiment
from garage.misc.instrument import variant
from garage.misc.instrument import VariantGenerator
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy


class VG(VariantGenerator):
    @variant
    def step_size(self):
        return [0.01, 0.05, 0.1]

    @variant
    def seed(self):
        return [1, 11, 21, 31, 41]


def run_task(vv):

    env = TfEnv(normalize(gym.make('HalfCheetah-v1')))

    policy = GaussianMLPPolicy(
        env_spec=env.spec, hidden_sizes=(32, 32), name="policy")

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=40,
        discount=0.99,
        step_size=vv["step_size"],
        # Uncomment both lines (this and the plot parameter below) to enable
        # plotting
        # plot=True,
    )
    algo.train()


variants = VG().variants()

for v in variants:

    run_experiment(
        run_task,
        exp_prefix="first_exp",
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a
        # random seed will be used
        seed=v["seed"],
        # mode="local",
        mode="ec2",
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )
    sys.exit()
