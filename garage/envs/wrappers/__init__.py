"""
gym.Env wrappers.

Used to transform an environment in a modular way.
It is also possible to apply multiple wrappers at the same
time.

Example:
    StackFrames(GrayScale(gym.make('env')))
"""

from garage.envs.wrappers.grayscale import Grayscale
from garage.envs.wrappers.repeat_action import RepeatAction
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.stack_frames import StackFrames

__all__ = ["RepeatAction", "Resize", "StackFrames", "Grayscale"]
