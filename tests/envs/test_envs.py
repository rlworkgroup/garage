import os

from nose2 import tools
MUJOCO_ENABLED = True
try:
    import mujoco_py
except OSError:
    print("Warning: Mujoco not installed. Skipping mujoco-related tests")
    MUJOCO_ENABLED = False
import numpy as np

from rllab.envs import DelayedActionEnv
from rllab.envs import GridWorldEnv
from rllab.envs import GymEnv
from rllab.envs import IdentificationEnv
from rllab.envs import NoisyObservationEnv
from rllab.envs import NormalizedEnv
from rllab.envs import ProxyEnv

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
envs.append(ProxyEnv(envs[0]))
envs.append(IdentificationEnv(CartpoleEnv, {}))
envs.append(NoisyObservationEnv(CartpoleEnv()))
envs.append(DelayedActionEnv(CartpoleEnv()))
envs.append(NormalizedEnv(CartpoleEnv()))


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
    env.close()
