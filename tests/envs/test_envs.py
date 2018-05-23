import numpy as np
from nose2 import tools

from rllab.envs.box2d import CarParkingEnv
from rllab.envs.box2d import CartpoleEnv
from rllab.envs.box2d import CartpoleSwingupEnv
from rllab.envs.box2d import DoublePendulumEnv
from rllab.envs.box2d import MountainCarEnv
from rllab.envs import GridWorldEnv
from rllab.envs import IdentificationEnv
import os

MUJOCO_ENABLED = True

try:
    import rllab.mujoco_py
    from rllab.envs.mujoco import HalfCheetahEnv
    from rllab.envs.mujoco import HopperEnv
    from rllab.envs.mujoco import InvertedDoublePendulumEnv
    from rllab.envs.mujoco import PointEnv
    from rllab.envs.mujoco import SimpleHumanoidEnv
    from rllab.envs.mujoco import SwimmerEnv
    from rllab.envs.mujoco import Walker2DEnv
    from rllab.envs.mujoco.gather import PointGatherEnv
    from rllab.envs.mujoco.gather import SwimmerGatherEnv
    from rllab.envs.mujoco.gather import AntGatherEnv
    from rllab.envs.mujoco.maze import PointMazeEnv
    from rllab.envs.mujoco.maze import SwimmerMazeEnv
    from rllab.envs.mujoco.maze import AntMazeEnv
except OSError:
    print("Warning: Mujoco not installed. Skipping mujoco-related tests")
    MUJOCO_ENABLED = False

from rllab.envs import NoisyObservationEnv, DelayedActionEnv
from rllab.envs import NormalizedEnv
from rllab.envs import ProxyEnv
from rllab.envs import GymEnv

simple_env_classes = [
    GridWorldEnv,
    CartpoleEnv,
    CarParkingEnv,
    CartpoleSwingupEnv,
    DoublePendulumEnv,
    MountainCarEnv,
]

if MUJOCO_ENABLED:
    simple_env_classes.extend([
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
        PointMazeEnv,
        SwimmerMazeEnv,
        AntMazeEnv,
    ])

envs = [cls() for cls in simple_env_classes]
envs.append(
    ProxyEnv(envs[0])
)
envs.append(
    IdentificationEnv(CartpoleEnv, {})
)
envs.append(
    NoisyObservationEnv(CartpoleEnv())
)
envs.append(
    DelayedActionEnv(CartpoleEnv())
)
envs.append(
    NormalizedEnv(CartpoleEnv())
)
envs.append(
    GymEnv('CartPole-v0')
)


@tools.params(*envs)
def test_env(env):
    print("Testing", env.__class__)
    ob_space = env.observation_space
    act_space = env.action_space
    ob = env.reset()
    assert ob_space.contains(ob)
    a = act_space.sample()
    assert act_space.contains(a)
    res = env.step(a)
    assert ob_space.contains(res.observation)
    assert np.isscalar(res.reward)
    if 'CIRCLECI' in os.environ:
        print("Skipping rendering test")
    else:
        env.render()
    env.terminate()
