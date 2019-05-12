from tests.fixtures.envs.dummy.base import DummyEnv
from tests.fixtures.envs.dummy.dummy_box_env import DummyBoxEnv
from tests.fixtures.envs.dummy.dummy_dict_env import DummyDictEnv
from tests.fixtures.envs.dummy.dummy_discrete_2d_env import DummyDiscrete2DEnv
from tests.fixtures.envs.dummy.dummy_discrete_env import DummyDiscreteEnv
from tests.fixtures.envs.dummy.dummy_discrete_pixel_env import (
    DummyDiscretePixelEnv)
from tests.fixtures.envs.dummy.dummy_discrete_pixel_env_baselines import (
    DummyDiscretePixelEnvBaselines)
from tests.fixtures.envs.dummy.dummy_reward_box_env import DummyRewardBoxEnv

__all__ = [
    'DummyEnv', 'DummyBoxEnv', 'DummyDictEnv', 'DummyDiscrete2DEnv',
    'DummyDiscreteEnv', 'DummyDiscretePixelEnv',
    'DummyDiscretePixelEnvBaselines', 'DummyRewardBoxEnv'
]
