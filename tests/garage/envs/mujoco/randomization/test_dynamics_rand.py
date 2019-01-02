import unittest

import numpy as np

from garage.envs.mujoco.randomization import Distribution
from garage.envs.mujoco.randomization import Method
from garage.envs.mujoco.randomization import randomize
from garage.envs.mujoco.randomization import Variations
from garage.envs.mujoco.swimmer_env import SwimmerEnv


class TestDynamicsRand(unittest.TestCase):
    def setUp(self):
        self._env = SwimmerEnv()
        self._bodyname = "mid"
        self._attribute = "pos"

        # values for UNIFORM Distributions
        self._lower_bound = np.array([0.5, 0.5, 0.5])
        self._upper_bound = np.array([1.5, 1.5, 1.5])
        # values for GAUSSIAN Distributions
        self._mean = self._lower_bound
        self._std_dev = self._lower_bound * 2

        # ensure you have legit bounds for randomization
        assert all(self._lower_bound != 1)
        assert all(self._upper_bound != 1)
        assert all(self._upper_bound > self._lower_bound)

        # Get original value for scaling
        body_id = self._env.sim.model._body_name2id[self._bodyname]
        self._orig_val = np.array(self._env.sim.model.body_pos[body_id])

    def test_absolute_method(self):
        # Create variation with absolute method of sampling
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        randomized_env = randomize(self._env, variations)
        randomized_vals = []

        for i in range(5):
            randomized_env.reset()
            body_id = randomized_env.wrapped_env.sim.model._body_name2id[
                self._bodyname]
            randomized_val = np.array(
                randomized_env.wrapped_env.sim.model.body_pos[body_id])
            randomized_vals.append(randomized_val)

            # check if within range
            assert all(self._lower_bound <= randomized_val), (
                "Randomised {1} < lower bound {0}".format(
                    self._lower_bound, randomized_val))
            assert all(randomized_val <= self._upper_bound), (
                "Randomised {1} > upper bound {0}".format(
                    self._upper_bound, randomized_val))

        # check that you have actual variation
        randomized_vals = np.array(randomized_vals)
        assert np.std(randomized_vals) > 0, ("Std Dev of randomized values "
                                             "not > 0. Getting the exact "
                                             "same numbers?\n {0}"
                                             .format(randomized_vals))

    def test_scaled_method(self):
        # Create variation with absolute method of sampling
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.SCALED) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        randomized_env = randomize(self._env, variations)
        randomized_vals = []

        for i in range(5):
            randomized_env.reset()
            body_id = randomized_env.wrapped_env.sim.model._body_name2id[
                self._bodyname]
            randomized_val = np.array(
                randomized_env.wrapped_env.sim.model.body_pos[body_id])
            randomized_vals.append(randomized_val)

            # check if within range
            assert all(self._orig_val * self._lower_bound <= randomized_val), (
                "Randomised {1} < lower bound {0}".format(
                    self._orig_val * self._lower_bound, randomized_val))
            assert all(randomized_val <= self._orig_val * self._upper_bound), (
                "Randomised {1} > upper bound {0}".format(
                    self._orig_val * self._upper_bound, randomized_val))

        # check that you have actual variation
        randomized_vals = np.array(randomized_vals)
        assert np.std(randomized_vals) > 0, ("Std Dev of randomized values "
                                             "not > 0. Getting the exact "
                                             "same numbers?\n {0}"
                                             .format(randomized_vals))

    def test_env_step(self):
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        second_var = "torso"  # Add a second
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(second_var)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        randomized_env = randomize(self._env, variations)

        # step() shouldn't error out (so it actually steps the wrapped env)
        for j in range(5):
            randomized_env.step(randomized_env.action_space.sample())

    def test_xml_node_exception(self):
        fake_node = "fake_node"
        variations = Variations()

        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(fake_node)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()
        with self.assertRaises(ValueError) as context:
            randomize(self._env, variations)

        assert fake_node in str(context.exception)

    def test_xml_attrib_exception(self):
        fake_attrib = "fake_attrib"
        variations = Variations()

        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(fake_attrib)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()
        with self.assertRaises(ValueError) as context:
            randomize(self._env, variations)

        assert fake_attrib in str(context.exception)

    def test_exception_gaussian_dist(self):
        variations = Variations()
        with self.assertRaises(ValueError) as context:
            variations.randomize() \
                .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
                .attribute("{0}".format(self._attribute)) \
                .with_method(Method.ABSOLUTE) \
                .sampled_from(Distribution.GAUSSIAN) \
                .with_range(self._lower_bound, self._upper_bound) \
                .add()
        assert "Need to call with_mean_std when sampled from Gaussian" \
               in str(context.exception)

    def test_exception_bad_distribution(self):
        # Use unused distribution Enum value for test
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(3) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        randomized_env = randomize(self._env, variations)

        with self.assertRaises(ValueError) as context:
            randomized_env.reset()
        assert "Unknown distribution" in str(context.exception)

    def test_exception_uniform_dist(self):
        variations = Variations()
        with self.assertRaises(ValueError) as context:
            variations.randomize() \
                .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
                .attribute("{0}".format(self._attribute)) \
                .with_method(Method.ABSOLUTE) \
                .sampled_from(Distribution.UNIFORM) \
                .with_mean_std(self._lower_bound, self._upper_bound) \
                .add()
        print(context.exception)
        assert "Need to call with_range when sampled from Uniform" \
               in str(context.exception)

    def test_exception_bad_sampling_method(self):
        # Use unused distribution Enum value for test
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(3) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        randomized_env = randomize(self._env, variations)

        with self.assertRaises(ValueError) as context:
            randomized_env.reset()
        assert "Unknown method" in str(context.exception)

    def test_prop_elem(self):
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        assert variations.get_list()[0].elem is None

    def test_prop_get_list(self):
        variations = Variations()
        assert variations.get_list() == []

    def test_prop_default(self):
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        assert variations.get_list()[0].default is None

    def test_prop_mean_std(self):
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.GAUSSIAN) \
            .with_mean_std(self._mean, self._std_dev) \
            .add()

        assert variations.get_list()[0].mean_std == (self._mean, self._std_dev)
