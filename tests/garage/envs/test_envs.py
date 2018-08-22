import unittest

import gym
from nose2 import tools
import numpy as np

from garage.envs import DelayedActionEnv
from garage.envs import GridWorldEnv
from garage.envs import IdentificationEnv
from garage.envs import NoisyObservationEnv
from garage.envs import NormalizedEnv
from garage.envs.box2d import CarParkingEnv
from garage.envs.box2d import CartpoleEnv
from garage.envs.box2d import CartpoleSwingupEnv
from garage.envs.box2d import DoublePendulumEnv
from garage.envs.box2d import MountainCarEnv

MUJOCO_ENABLED = True
try:
    from garage.envs.mujoco import HalfCheetahEnv
    from garage.envs.mujoco import HopperEnv
    from garage.envs.mujoco import InvertedDoublePendulumEnv
    from garage.envs.mujoco import PointEnv
    from garage.envs.mujoco import SimpleHumanoidEnv
    from garage.envs.mujoco import SwimmerEnv
    from garage.envs.mujoco import Walker2DEnv
    from garage.envs.mujoco.gather import AntGatherEnv
    from garage.envs.mujoco.gather import PointGatherEnv
    from garage.envs.mujoco.gather import SwimmerGatherEnv
    from garage.envs.mujoco.maze import AntMazeEnv
    from garage.envs.mujoco.maze import PointMazeEnv
    from garage.envs.mujoco.maze import SwimmerMazeEnv
except OSError:
    print("Warning: MuJoCo not installed. Skipping MuJoCo-related tests")
    MUJOCO_ENABLED = False

simple_env_classes = [
    GridWorldEnv,
    CartpoleEnv,
    CarParkingEnv,
    CartpoleSwingupEnv,
    DoublePendulumEnv,
    MountainCarEnv,
]

mujoco_classes = [
    SwimmerMazeEnv,
    PointMazeEnv,
    AntMazeEnv,
]

mujoco_rgb_classes = [
    PointEnv,
    Walker2DEnv,
    SwimmerEnv,
    SimpleHumanoidEnv,
    InvertedDoublePendulumEnv,
    HopperEnv,
    HalfCheetahEnv,
    PointGatherEnv,
    SwimmerGatherEnv,
    AntGatherEnv,
]

classes = []
classes.extend(simple_env_classes)

if MUJOCO_ENABLED:
    classes.extend(mujoco_classes)
    classes.extend(mujoco_rgb_classes)

# Call constructor for each class
envs = [cls() for cls in classes]

cartpole_envs = [
    IdentificationEnv(CartpoleEnv, {}),
    NoisyObservationEnv(CartpoleEnv()),
    DelayedActionEnv(CartpoleEnv()),
    NormalizedEnv(CartpoleEnv()),
    gym.make("CartPole-v1"),
]
envs.extend(cartpole_envs)


class TestEnvs(unittest.TestCase):
    @tools.params(*envs)
    def test_env(self, env):
        print("Testing", env.__class__)
        ob_space = env.observation_space
        act_space = env.action_space
        ob = env.reset()
        assert ob_space.contains(ob)
        a = act_space.sample()
        assert act_space.contains(a)
        res = env.step(a)
        assert ob_space.contains(res[0])  # res[0] --> observation
        assert np.isscalar(res[1])  # res[1] --> reward
        if any(isinstance(env, T) for T in mujoco_rgb_classes):
            img = env.render(mode="rgb_array")
            if img is None:
                img = env.render(mode="human")
            assert img is not None
            assert img.shape[0] == env.render_width
            assert img.shape[1] == env.render_height
        else:
            env.render()
        env.close()
