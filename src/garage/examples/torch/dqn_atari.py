#!/usr/bin/env python3
"""An example to train a task with DQN algorithm.

Here it creates a gym environment CartPole, and trains a DQN with 50k steps.
"""
import math

import click
import gym
import numpy as np
import psutil
import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.envs.wrappers.clip_reward import ClipReward
from garage.envs.wrappers.episodic_life import EpisodicLife
from garage.envs.wrappers.fire_reset import FireReset
from garage.envs.wrappers.grayscale import Grayscale
from garage.envs.wrappers.max_and_skip import MaxAndSkip
from garage.envs.wrappers.noop import Noop
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.stack_frames import StackFrames
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import EpsilonGreedyPolicy
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import DQN
from garage.torch.policies import DiscreteQFArgmaxPolicy
from garage.torch.q_functions import DiscreteCNNQFunction
from garage.trainer import Trainer

hyperparams = dict(n_epochs=500,
                   steps_per_epoch=20,
                   sampler_batch_size=500,
                   lr=1e-4,
                   discount=0.99,
                   min_buffer_size=int(1e4),
                   n_train_steps=125,
                   target_update_freq=2,
                   buffer_batch_size=32,
                   max_epsilon=1.0,
                   min_epsilon=0.01,
                   decay_ratio=0.1,
                   buffer_size=int(1e4),
                   hidden_sizes=(512, ),
                   hidden_channels=(32, 64, 64),
                   kernel_sizes=(8, 4, 3),
                   strides=(4, 2, 1),
                   clip_gradient=10)


@click.command()
@click.argument('env', type=str)
@click.option('--seed', default=24)
@click.option('--n', type=int, default=psutil.cpu_count(logical=False))
@click.option('--buffer_size', type=int, default=None)
@click.option('--n_steps', type=float, default=None)
@click.option('--max_episode_length', type=int, default=None)
def main(env=None,
         seed=24,
         n=psutil.cpu_count(logical=False),
         buffer_size=None,
         n_steps=None,
         max_episode_length=None):
    """Wrapper to setup the logging directory.

    Args:
        env (str): Name of the atari environment, can either be the prefix
            or the full name. For example, this can either be 'Pong' or
            'PongNoFrameskip-v4'. If the former is used, the env used will be
            `env` + 'NoFrameskip-v4'.
        seed (int): Seed to use for the RNG.
        n (int): Number of workers to use. Defaults to the number of CPU cores
            available.
        buffer_size (int): size of the replay buffer in transitions. If None,
            defaults to hyperparams['buffer_size']. This is used by the
            integration tests.
        n_steps (float): Total number of environment steps to run for, not
            not including evaluation. If this is not None, n_epochs will
            be recalculated based on this value.
        max_episode_length (int): Max length of an episode. If None, defaults
            to the timelimit specific to the environment. Used by integration
            tests.
    """
    if '-v' not in env:
        env += 'NoFrameskip-v4'
    logdir = 'data/local/experiment/' + env

    if n_steps is not None:
        hyperparams['n_epochs'] = math.ceil(
            int(n_steps) / (hyperparams['steps_per_epoch'] *
                            hyperparams['sampler_batch_size']))
    if buffer_size is not None:
        hyperparams['buffer_size'] = buffer_size

    dqn_atari(dict(log_dir=logdir),
              env=env,
              seed=seed,
              n_workers=n,
              max_episode_length=max_episode_length,
              **hyperparams)


# pylint: disable=unused-argument
@wrap_experiment(snapshot_mode='gap_overwrite', snapshot_gap=30)
def dqn_atari(ctxt=None,
              env=None,
              seed=24,
              n_workers=psutil.cpu_count(logical=False),
              max_episode_length=None,
              **kwargs):
    """Train DQN with PongNoFrameskip-v4 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        env (str): Name of the atari environment, eg. 'PongNoFrameskip-v4'.
        seed (int): Used to seed the random number generator to produce
            determinism.
        n_workers (int): Number of workers to use. Defaults to the number of
            CPU cores available.
        max_episode_length (int): Max length of an episode. If None, defaults
            to the timelimit specific to the environment. Used by integration
            tests.
        kwargs (dict): hyperparameters to be saved to variant.json.

    """
    assert n_workers > 0
    assert env is not None
    env = gym.make(env)
    env = Noop(env, noop_max=30)
    env = MaxAndSkip(env, skip=4)
    env = EpisodicLife(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireReset(env)
    env = Grayscale(env)
    env = Resize(env, 84, 84)
    env = ClipReward(env)
    env = StackFrames(env, 4, axis=0)
    env = GymEnv(env, max_episode_length=max_episode_length, is_image=True)
    set_seed(seed)
    trainer = Trainer(ctxt)

    n_epochs = hyperparams['n_epochs']
    steps_per_epoch = hyperparams['steps_per_epoch']
    sampler_batch_size = hyperparams['sampler_batch_size']
    num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size
    replay_buffer = PathBuffer(
        capacity_in_transitions=hyperparams['buffer_size'])

    qf = DiscreteCNNQFunction(
        env_spec=env.spec,
        image_format='NCHW',
        hidden_channels=hyperparams['hidden_channels'],
        kernel_sizes=hyperparams['kernel_sizes'],
        strides=hyperparams['strides'],
        hidden_w_init=(
            lambda x: torch.nn.init.orthogonal_(x, gain=np.sqrt(2))),
        hidden_sizes=hyperparams['hidden_sizes'])

    policy = DiscreteQFArgmaxPolicy(env_spec=env.spec, qf=qf)
    exploration_policy = EpsilonGreedyPolicy(
        env_spec=env.spec,
        policy=policy,
        total_timesteps=num_timesteps,
        max_epsilon=hyperparams['max_epsilon'],
        min_epsilon=hyperparams['min_epsilon'],
        decay_ratio=hyperparams['decay_ratio'])

    sampler = LocalSampler(agents=exploration_policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker,
                           n_workers=n_workers)

    algo = DQN(env_spec=env.spec,
               policy=policy,
               qf=qf,
               exploration_policy=exploration_policy,
               replay_buffer=replay_buffer,
               sampler=sampler,
               steps_per_epoch=steps_per_epoch,
               qf_lr=hyperparams['lr'],
               clip_gradient=hyperparams['clip_gradient'],
               discount=hyperparams['discount'],
               min_buffer_size=hyperparams['min_buffer_size'],
               n_train_steps=hyperparams['n_train_steps'],
               target_update_freq=hyperparams['target_update_freq'],
               buffer_batch_size=hyperparams['buffer_batch_size'])

    set_gpu_mode(False)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        set_gpu_mode(True)
        algo.to()

    trainer.setup(algo, env)

    trainer.train(n_epochs=n_epochs, batch_size=sampler_batch_size)
    env.close()


main()
