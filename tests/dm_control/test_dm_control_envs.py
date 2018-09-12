import pickle

from dm_control.suite import ALL_TASKS
import numpy as np

from garage.envs.dm_control import DmControlEnv
from tests.fixtures import GarageTestCase


class TestDmControlEnvs(GarageTestCase):
    def test_dm_control_envs(self):
        for task in ALL_TASKS:
            env = DmControlEnv(domain_name=task[0], task_name=task[1])

            # Test pickling
            pickle.loads(pickle.dumps(env))

            print("Testing", env.__class__)
            ob_space = env.observation_space
            act_space = env.action_space
            ob = env.reset()
            assert ob_space.contains(ob)
            a = act_space.sample()
            assert act_space.contains(a)
            for _ in range(10):
                res = env.step(a)
                assert ob_space.contains(res[0])  # res[0] --> observation
                assert np.isscalar(res[1])  # res[1] --> reward
                env.render()
            env.close()

            del env
