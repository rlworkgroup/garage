"""
This is an example to train a task with DQN algorithm in pixel environment.

Here it creates a gym environment Pong, and trains a DQN with 1M steps.
"""
import gym

from garage.envs.wrappers.clip_reward import ClipReward
from garage.envs.wrappers.episodic_life import EpisodicLife
from garage.envs.wrappers.fire_reset import FireReset
from garage.envs.wrappers.grayscale import Grayscale
from garage.envs.wrappers.max_and_skip import MaxAndSkip
from garage.envs.wrappers.noop import Noop
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.stack_frames import StackFrames
from garage.experiment import LocalRunner, run_experiment
from garage.np.exploration_strategies import EpsilonGreedyStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DQN
from garage.tf.envs import TfEnv
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteCNNQFunction


def run_task(*_):
    """Run task."""
    with LocalRunner() as runner:
        n_epochs = 100
        n_epoch_cycles = 20
        sampler_batch_size = 500
        num_timesteps = n_epochs * n_epoch_cycles * sampler_batch_size
        env = gym.make('PongNoFrameskip-v4')
        env = Noop(env, noop_max=30)
        env = MaxAndSkip(env, skip=4)
        env = EpisodicLife(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireReset(env)
        env = Grayscale(env)
        env = Resize(env, 84, 84)
        env = ClipReward(env)
        env = StackFrames(env, 4)

        env = TfEnv(env)

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec, size_in_transitions=int(5e4), time_horizon=1)

        qf = DiscreteCNNQFunction(
            env_spec=env.spec,
            filter_dims=(8, 4, 3),
            num_filters=(32, 64, 64),
            strides=(4, 2, 1),
            dueling=False)

        policy = DiscreteQfDerivedPolicy(env_spec=env.spec, qf=qf)
        epilson_greedy_strategy = EpsilonGreedyStrategy(
            env_spec=env.spec,
            total_timesteps=num_timesteps,
            max_epsilon=1.0,
            min_epsilon=0.02,
            decay_ratio=0.1)

        algo = DQN(
            env_spec=env.spec,
            policy=policy,
            qf=qf,
            exploration_strategy=epilson_greedy_strategy,
            replay_buffer=replay_buffer,
            qf_lr=1e-4,
            discount=0.99,
            min_buffer_size=int(1e4),
            double_q=False,
            n_train_steps=500,
            n_epoch_cycles=n_epoch_cycles,
            target_network_update_freq=2,
            buffer_batch_size=32)

        runner.setup(algo, env)
        runner.train(
            n_epochs=n_epochs,
            n_epoch_cycles=n_epoch_cycles,
            batch_size=sampler_batch_size)


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode='last',
    seed=1,
    plot=False,
)
