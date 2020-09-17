import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import click
import gym
import torch
from garage.experiment import Snapshotter
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from garage.envs import GymEnv
import numpy as np
from garage import rollout

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
from garage.sampler import DefaultWorker, LocalSampler, RaySampler, FragmentWorker
from garage.torch import set_gpu_mode
from garage.torch.algos import DQN
from garage.torch.policies import DiscreteQFDerivedPolicy
from garage.torch.q_functions import DiscreteCNNQFunction


deepmind = wrap_deepmind(make_atari('BeamRiderNoFrameskip-v4'), frame_stack=True)


env = gym.make('BeamRiderNoFrameskip-v4')
env = Noop(env, noop_max=30)
env = MaxAndSkip(env, skip=4)
env = EpisodicLife(env)
if 'FIRE' in env.unwrapped.get_action_meanings():
    env = FireReset(env)
env = Grayscale(env)
env = Resize(env, 84, 84)
env = ClipReward(env)
env = StackFrames(env, 4)

obs = env.reset()
obs = np.expand_dims(obs.reshape((4, 84, 84)), 0).reshape(1, 84, 84, 4)[0]
# obs = np.moveaxis(obs, 0, 2)
plt.imshow(obs, cmap='gray', vmin=0, vmax=255) #grayscale colormap
plt.show()

obs = deepmind.reset()
obs = np.asarray(obs)
print(obs.shape)
obs = obs.reshape((84, 84, 4))
# obs = np.moveaxis(obs, 0, 2)
plt.imshow(obs, cmap='gray', vmin=0, vmax=255) #grayscale colormap
plt.show()
# obs_shape = ((len(observations), ) +
#     self._env_spec.observation_space.shape)
# observations = observations.reshape(obs_shape)


# obs = observations[0].cpu().numpy()
# new_shape = (84, 84, 4)
# obs = obs.reshape(new_shape)
# obs = obs[:,:,:-1].astype(int)
# print(obs.shape)
# plt.imshow(obs, cmap='gray', vmin=0, vmax=255) #grayscale colormap
# plt.show()
# print('DONE')