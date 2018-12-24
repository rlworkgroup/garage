import unittest

import numpy as np

from garage.envs.mujoco.randomization import Distribution
from garage.envs.mujoco.randomization import Method
from garage.envs.mujoco.randomization import randomize
from garage.envs.mujoco.randomization import Variations
from garage.envs.mujoco.swimmer_env import SwimmerEnv


class TestDynamicsRand(unittest.TestCase):
    def setUp(self):
        self._bodyname = "mid"
        self._attribute = "pos"
        self._lower_bound = np.array([0.5, 0.5, 0.5])
        self._upper_bound = np.array([1.5, 1.5, 1.5])
        self._env = SwimmerEnv()

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
        variations.randomize()\
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname))\
            .attribute("{0}".format(self._attribute))\
            .with_method(Method.ABSOLUTE)\
            .sampled_from(Distribution.UNIFORM)\
            .with_range(self._lower_bound, self._upper_bound)\
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
        variations.randomize()\
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname))\
            .attribute("{0}".format(self._attribute))\
            .with_method(Method.SCALED)\
            .sampled_from(Distribution.UNIFORM)\
            .with_range(self._lower_bound, self._upper_bound)\
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

        randomized_env = randomize(self._env, variations)

        # step() shouldn't error out (so it actually steps the wrapped env)
        for j in range(5):
            randomized_env.step(randomized_env.action_space.sample())
