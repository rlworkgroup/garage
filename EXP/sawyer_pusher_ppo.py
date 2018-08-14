"""EXP to train a sawyer pusher with PPO algorithm."""

from types import SimpleNamespace

from garage.envs import normalize
from garage.envs.mujoco.sawyer.push_env import SimplePushEnv
from garage.misc.instrument import run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
# from garage.tf.policies import GaussianMLPPolicy
from sandbox.embed2learn.policies import GaussianMLPPolicy


def run_task(v):
    v = SimpleNamespace(**v)

    env = TfEnv(normalize(SimplePushEnv(control_method='position_control')))

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(256, 128),
        std_share_network=True,
        init_std=v.policy_init_std,)

    baseline = GaussianMLPBaseline(env_spec=env.spec)

    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v.batch_size,
        max_path_length=v.max_path_length,
        n_itr=10000,
        discount=0.99,
        step_size=0.2,
        optimizer_args=dict(batch_size=32, max_epochs=10),
        plot=True)
    algo.train()


config = dict(
    batch_size=16384,
    max_path_length=500,
    policy_init_std=1.0
)


run_experiment(
    run_task,
    exp_prefix='sawyer_push_ppo_position',
    n_parallel=12,
    snapshot_mode="last",
    variant=config,
    seed=1,
    plot=True,
)
