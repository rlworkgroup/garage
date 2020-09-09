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

from baselines.common.atari_wrappers import make_atari, wrap_deepmind

import gym
import torch

import gym
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from gym.utils.play import play


import time
env = gym.make('PongNoFrameskip-v4')
env = Noop(env, noop_max=30)
env = MaxAndSkip(env, skip=4)
env = EpisodicLife(env)
if 'FIRE' in env.unwrapped.get_action_meanings():
    env = FireReset(env)
env = Grayscale(env)
env = Resize(env, 84, 84)
env = ClipReward(env)
env = StackFrames(env, 3)

# env = make_atari('PongNoFrameskip-v4')
# env = wrap_deepmind(env, frame_stack=False, scale=True)

play(env)