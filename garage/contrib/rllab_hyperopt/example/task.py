from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs import PointEnv
from garage.tf.algos import TRPO
from garage.tf.policies import GaussianMLPPolicy


def run_task(v):
    env = normalize(PointEnv())

    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

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
        # plot=True,
    )
    algo.train()
