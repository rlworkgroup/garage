import gym

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.misc.instrument import run_experiment
from garage.theano.algos import TRPO
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy


def run_task(*_):
    env = TheanoEnv(normalize(gym.make("Pendulum-v0")))

    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

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
