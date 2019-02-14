"""This script tests garage.tf.spaces.dict functionality."""
import unittest

from garage.misc import ext
from garage.tf.envs import TfEnv
from tests.fixtures.envs.dummy import DummyDictEnv


class TestDictSpace(unittest.TestCase):
    def test_dict_space(self):
        ext.set_seed(0)

        # A dummy dict env
        dummy_env = DummyDictEnv()
        dummy_act = dummy_env.action_space
        dummy_act_sample = dummy_act.sample()

        # A dummy dict env wrapped by garage.tf
        tf_env = TfEnv(dummy_env)
        tf_act = tf_env.action_space
        tf_obs = tf_env.observation_space

        # flat_dim
        assert tf_act.flat_dim == tf_act.flatten(dummy_act_sample).shape[-1]

        # flat_dim_with_keys
        assert tf_obs.flat_dim == tf_obs.flat_dim_with_keys(
            iter(["achieved_goal", "desired_goal", "observation"]))

        # un/flatten
        assert tf_act.unflatten(
            tf_act.flatten(dummy_act_sample)) == dummy_act_sample

        # un/flatten_n
        samples = [dummy_act.sample() for _ in range(10)]
        assert tf_act.unflatten_n(tf_act.flatten_n(samples)) == samples

        # un/flatten_with_keys
        assert tf_act.unflatten_with_keys(
            tf_act.flatten_with_keys(dummy_act_sample, iter(["action"])),
            iter(["action"]))

        tf_env.close()
