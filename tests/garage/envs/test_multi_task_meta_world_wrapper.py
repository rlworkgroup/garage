"""Tests for the module MTMetaWorldWrapper."""
from metaworld.benchmarks import MT10
import pytest

from garage.envs import MTMetaWorldWrapper


@pytest.mark.mujoco
class TestMTMetaWorldWrapper:
    """Tests for the module MTMetaWorldWrapper."""

    def setup_class(self):
        """Init Wrapper with MT10."""
        tasks = MT10.get_train_tasks().all_task_names
        envs = []
        for task in tasks:
            envs.append(MT10.from_task(task))
        self.env = MTMetaWorldWrapper(envs)
        self.mt10_env = MT10.get_train_tasks()

    def teardown_class(self):
        """Close the MTMetaWorldWrapper."""
        self.env.close()

    def test_num_tasks(self):
        """Assert num tasks returns 10, because MT10 is being tested."""
        assert self.env.num_tasks == 10

    def test_task_names(self):
        """Assert task names are the same across MT10 and wrapped envs."""
        assert set(self.env.task_names) == set(self.mt10_env.all_task_names)

    def test_step(self):
        """Test that env_infos includes extra infos and obs has onehot."""
        action = self.env.spec.action_space.sample()
        obs, _, _, info = self.env.step(action)
        assert info['task_id'] == self.env.active_task_id
        assert info['task_name'] == self.env.active_task_name
        assert (self.env.active_task_one_hot == obs[9:]).all()

    def test_reset(self):
        """Test round robin switching of environments during call to reset."""
        self.env.reset()
        active_task_id = self.env.active_task_id
        for _ in range(self.env.num_tasks):
            self.env.reset()
            action = self.env.spec.action_space.sample()
            _, _, _, info = self.env.step(action)
            assert not info['task_id'] == active_task_id
            active_task_id = self.env.active_task_id
