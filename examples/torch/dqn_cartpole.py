#!/usr/bin/env python3
"""An example to train a task with DQN algorithm.

Here it creates a gym environment CartPole, and trains a DQN with 50k steps.
"""
import click

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import EpsilonGreedyPolicy
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch.algos import DQN
from garage.torch.policies import DiscreteQFArgmaxPolicy
from garage.torch.q_functions import DiscreteMLPQFunction
from garage.trainer import Trainer


@click.command()
@click.option('--seed', default=24)
@wrap_experiment(snapshot_mode='none')
def dqn_cartpole(ctxt=None, seed=24):
    """Train DQN with CartPole-v0 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)
    runner = Trainer(ctxt)

    n_epochs = 100
    steps_per_epoch = 10
    sampler_batch_size = 512
    num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size
    env = GymEnv('CartPole-v0')
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
    qf = DiscreteMLPQFunction(env_spec=env.spec, hidden_sizes=(8, 5))
    policy = DiscreteQFArgmaxPolicy(env_spec=env.spec, qf=qf)
    exploration_policy = EpsilonGreedyPolicy(env_spec=env.spec,
                                             policy=policy,
                                             total_timesteps=num_timesteps,
                                             max_epsilon=1.0,
                                             min_epsilon=0.01,
                                             decay_ratio=0.4)
    sampler = LocalSampler(agents=exploration_policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker)
    algo = DQN(env_spec=env.spec,
               policy=policy,
               qf=qf,
               exploration_policy=exploration_policy,
               replay_buffer=replay_buffer,
               sampler=sampler,
               steps_per_epoch=steps_per_epoch,
               qf_lr=5e-5,
               discount=0.9,
               min_buffer_size=int(1e4),
               n_train_steps=500,
               target_update_freq=30,
               buffer_batch_size=64)

    runner.setup(algo, env)
    runner.train(n_epochs=n_epochs, batch_size=sampler_batch_size)

    env.close()


dqn_cartpole()
