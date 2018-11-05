import gym

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.experiment import run_experiment
from garage.theano.algos import TRPO
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy


def run_task(*_):
    # Please note that different environments with different action spaces may
    # require different policies. For example with a Box action space, a
    # GaussianMLPPolicy works, but for a Discrete action space may need to use
    # a CategoricalMLPPolicy (see the trpo_gym_cartpole.py example)
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
        # Uncomment both lines (this and the plot parameter below) to enable
        # plotting
        # plot=True,
    )
    algo.train()


run_experiment(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random
    # seed will be used
    seed=1,
    # plot=True,
)
