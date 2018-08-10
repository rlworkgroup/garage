"""This script tests garage.tf.spaces.dict functionality."""
import unittest

from tests.envs.dummy import DummyDictEnv

from garage.misc import ext
from garage.tf.envs import TfEnv


class TestDictSpace(unittest.TestCase):
    def test_dict_space(self):
        ext.set_seed(0)

        # A dummy dict env
        dummy_env = DummyDictEnv()
        dummy_act = dummy_env.action_space
        dummy_act_sample = dummy_act.sample()

        # A dummy dict env wrapped by garage.tf
        tf_env = TfEnv(DummyDictEnv())
        tf_act = tf_env.action_space
        tf_obs = tf_env.observation_space

        # sample
        assert tf_act.sample() == dummy_act_sample

        # flat_dim
        assert tf_act.flat_dim == tf_act.flatten(dummy_act_sample).shape[-1]

        # flat_dim_with_keys
        assert tf_obs.flat_dim == tf_obs.flat_dim_with_keys(
            iter(["achieved_goal", "desired_goal", "observation"]))

        # un/flatten
        assert tf_act.unflatten(
            tf_act.flatten(dummy_act_sample)) == dummy_act_sample

        # TODO: un/flatten_n

        # TODO: un/flatten_with_keys
