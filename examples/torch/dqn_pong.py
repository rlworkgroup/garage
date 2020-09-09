#!/usr/bin/env python3
"""An example to train a task with DQN algorithm.

Here it creates a gym environment CartPole, and trains a DQN with 50k steps.
"""
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
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import EpsilonGreedyPolicy
from garage.replay_buffer import PathBuffer
from garage.torch.algos import DQN
from garage.torch.policies import DiscreteQFDerivedPolicy
from garage.torch.q_functions import DiscreteCNNQFunction
from garage.torch import set_gpu_mode
from garage.sampler import DefaultWorker, RaySampler, LocalSampler

import gym
import torch
import click

hyperparams = dict(
    n_epochs=500,
    steps_per_epoch=20,
    sampler_batch_size=500, #10k
    lr=1e-4,
    discount=0.99,
    min_buffer_size=int(1e4), #deepmind 5e4
    n_train_steps=500,
    target_update_freq=20,
    buffer_batch_size=32,
    max_epsilon=1.0,
    min_epsilon=0.01,
    decay_ratio=0.1,
    buffer_size=int(1e4),
    hidden_sizes=(512, ),
)

@click.command()
@click.option('--seed', default=1)
@wrap_experiment(snapshot_mode='last', snapshot_gap=3)
def dqn_pong(ctxt=None, seed=24, **kwargs):
    """Train DQN with PongNoFrameskip-v4 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """

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
    env = GymEnv(env)
    set_seed(seed)
    runner = LocalRunner(ctxt)

    env.spec.observation_space = env.observation_space
    env.spec.action_space = env.action_space

    n_epochs = hyperparams['n_epochs']
    steps_per_epoch = hyperparams['steps_per_epoch']
    sampler_batch_size = hyperparams['sampler_batch_size']
    num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size
    replay_buffer = PathBuffer(
        capacity_in_transitions=hyperparams['buffer_size'])

    qf = DiscreteCNNQFunction(env_spec=env.spec,
                              minibatch_size=hyperparams['buffer_batch_size'],
                              hidden_channels=(32, 64, 64),
                              kernel_sizes=(8, 4, 3),
                              strides=(4, 2, 1),
                              hidden_sizes=hyperparams['hidden_sizes'],
                              is_image=True)

    policy = DiscreteQFDerivedPolicy(env_spec=env.spec, qf=qf)
    exploration_policy = EpsilonGreedyPolicy(
        env_spec=env.spec,
        policy=policy,
        total_timesteps=num_timesteps,
        max_epsilon=hyperparams['max_epsilon'],
        min_epsilon=hyperparams['min_epsilon'],
        decay_ratio=hyperparams['decay_ratio'])

    algo = DQN(env_spec=env.spec,
               policy=policy,
               qf=qf,
               exploration_policy=exploration_policy,
               replay_buffer=replay_buffer,
               steps_per_epoch=steps_per_epoch,
               qf_lr=hyperparams['lr'],
               clip_gradient=None,
               discount=hyperparams['discount'],
               min_buffer_size=hyperparams['min_buffer_size'],
               n_train_steps=hyperparams['n_train_steps'],
               target_update_freq=hyperparams['target_update_freq'],
               buffer_batch_size=hyperparams['buffer_batch_size'])

    if torch.cuda.is_available():
        set_gpu_mode(True)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        set_gpu_mode(False)
    algo.to()

    runner.setup(algo, env, sampler_cls=LocalSampler)
    runner.train(n_epochs=n_epochs, batch_size=sampler_batch_size)


dqn_pong()
