import pytest

from garage.envs.half_cheetah_dir_env import HalfCheetahDirEnv
from garage.envs.half_cheetah_vel_env import HalfCheetahVelEnv


class TestMetaHalfCheetahEnvs:

    @pytest.mark.parametrize('env_type',
                             [HalfCheetahVelEnv, HalfCheetahDirEnv])
    def test_can_sim(self, env_type):
        env = env_type()
        task = env.sample_tasks(1)[0]
        env.set_task(task)
        for _ in range(3):
            env.step(env.action_space.sample())
