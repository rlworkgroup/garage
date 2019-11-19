"""gym.Env wrappers.

Used to transform an environment in a modular way.
It is also possible to apply multiple wrappers at the same
time.

Example:
    StackFrames(GrayScale(gym.make('env')))

"""
from garage.envs.wrappers.atari_env import AtariEnv
from garage.envs.wrappers.clip_reward import ClipReward
from garage.envs.wrappers.episodic_life import EpisodicLife
from garage.envs.wrappers.fire_reset import FireReset
from garage.envs.wrappers.grayscale import Grayscale
from garage.envs.wrappers.max_and_skip import MaxAndSkip
from garage.envs.wrappers.noop import Noop
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.stack_frames import StackFrames

__all__ = [
    'AtariEnv', 'ClipReward', 'EpisodicLife', 'FireReset', 'Grayscale',
    'MaxAndSkip', 'Noop', 'Resize', 'StackFrames'
]
