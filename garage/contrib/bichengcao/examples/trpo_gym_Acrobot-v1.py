import gym

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.experiment import run_experiment
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy


def run_task(*_):
    env = TfEnv(normalize(gym.make("Acrobot-v1")))

    policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=env.max_episode_steps,
        n_itr=50,
        discount=0.99,
        step_size=0.01,
        plot=True,
    )
    algo.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    plot=True,
)
