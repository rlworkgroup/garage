from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.experiment import run_experiment
from garage.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.theano.algos import DDPG
from garage.theano.envs import TheanoEnv
from garage.theano.policies import DeterministicMLPPolicy
from garage.theano.q_functions import ContinuousMLPQFunction


def run_task(*_):
    env = TheanoEnv(normalize(CartpoleEnv()))

    policy = DeterministicMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers,
        # each with 32 hidden units.
        hidden_sizes=(32, 32))

    es = OUStrategy(env_spec=env.spec)

    qf = ContinuousMLPQFunction(env_spec=env.spec)

    replay_buffer = SimpleReplayBuffer(
        env_spec=env.spec, size_in_transitions=int(1e6), time_horizon=100)

    algo = DDPG(
        env=env,
        policy=policy,
        es=es,
        qf=qf,
        pool=replay_buffer,
        batch_size=32,
        max_path_length=100,
        epoch_length=1000,
        min_pool_size=10000,
        n_epochs=1000,
        discount=0.99,
        scale_reward=0.01,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-4,
        # Uncomment both lines (this and the plot parameter below) to enable
        # plotting
        plot=True,
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
