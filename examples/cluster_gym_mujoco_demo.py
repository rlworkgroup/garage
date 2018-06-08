import sys

from rllab.baselines import LinearFeatureBaseline
from rllab.envs import GymEnv
from rllab.envs import normalize
from rllab.envs.gym_util.env_util import spec
from rllab.misc.instrument import run_experiment_lite
from rllab.misc.instrument import variant
from rllab.misc.instrument import VariantGenerator
from rllab.tf.algos import TRPO
from rllab.tf.envs import TfEnv
from rllab.tf.policies import GaussianMLPPolicy


class VG(VariantGenerator):
    @variant
    def step_size(self):
        return [0.01, 0.05, 0.1]

    @variant
    def seed(self):
        return [1, 11, 21, 31, 41]


def run_task(vv):

    env = TfEnv(
        normalize(
            GymEnv('HalfCheetah-v1', record_video=False, record_log=False)))

    policy = GaussianMLPPolicy(
        env_spec=spec(env),
        hidden_sizes=(32, 32),
        name="policy")

    baseline = LinearFeatureBaseline(env_spec=spec(env))

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
        # plotting plot=True,
    )
    algo.train()


variants = VG().variants()

for v in variants:

    run_experiment_lite(
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
