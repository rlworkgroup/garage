import unittest
from os import path

import gym
import numpy as np

from garage.contrib.exp.checkpointers import DiskCheckpointer


class TestDiskCheckpointer(unittest.TestCase):
    def test_disk_checkpointer(self):
        exp_dir = "test_disk_checkpointer"
        prefix = "exp"

        foo = "TestString"
        bar = {
            "numpy_array": np.array([[1, 2, 3], [2, 3, 4]])
        }
        env = gym.make('Pendulum-v0')
        checkpointer = DiskCheckpointer(exp_dir, prefix=prefix)
        checkpointer._clean_outdated()

        objs = checkpointer.load(foo=foo, bar=bar, env=env)
        foo_ = objs["foo"]
        bar_ = objs["bar"]
        env_ = objs["env"]

        # Save initial checkpoint
        assert foo_ == foo and bar_ == bar and env_ == env

        foo = foo + "modified"
        bar["new_key"] = "new_value"
        env = gym.make('CartPole-v1')
        checkpointer.save(foo=foo, bar=bar, env=env)

        objs = checkpointer.load(foo=foo_, bar=bar_, env=env_)
        foo_ = objs["foo"]
        bar_ = objs["bar"]
        env_ = objs["env"]

        # Load from checkpointer
        assert foo_ == foo
        assert bar_["new_key"] == "new_value"
        assert np.array_equal(bar_["numpy_array"], bar["numpy_array"])
        assert env.action_space == env_.action_space

        checkpointer._clean_outdated()

        # Make sure checkpoint directory is removed
        assert not path.isdir(exp_dir)
