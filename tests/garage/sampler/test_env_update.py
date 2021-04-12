from garage.sampler import SetTaskUpdate

from tests.fixtures.envs.dummy import DummyBoxEnv

TEST_TASK = ['test_task']


class MTDummyEnv(DummyBoxEnv):

    def set_task(self, task):
        assert task == TEST_TASK


class MTDummyEnvSubtype(MTDummyEnv):
    pass


def test_set_task_update_with_subtype():
    old_env = MTDummyEnvSubtype()
    env_update = SetTaskUpdate(MTDummyEnv, TEST_TASK, None)
    new_env = env_update(old_env)
    assert new_env is not old_env
    assert new_env is not None
    assert old_env is not None
