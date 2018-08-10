# This doesn't work. After 150 iterations still didn't learn anything.
import gym

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.misc.instrument import run_experiment
from garage.theano.algos import TRPO
from garage.theano.envs import TheanoEnv
from garage.theano.policies import CategoricalMLPPolicy


def run_task(*_):
    env = TheanoEnv(normalize(gym.make("MountainCar-v0")))

    policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=env.max_episode_steps,
        n_itr=150,
        discount=0.99,
        step_size=0.1,
        plot=True,
    )
    algo.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    plot=True,
)
