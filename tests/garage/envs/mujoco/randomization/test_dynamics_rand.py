import unittest

import numpy as np

from garage.envs.mujoco.randomization import Distribution
from garage.envs.mujoco.randomization import Method
from garage.envs.mujoco.randomization import randomize
from garage.envs.mujoco.randomization import Variations
from garage.envs.mujoco.swimmer_env import SwimmerEnv


class TestDynamicsRand(unittest.TestCase):
    def test_dynamics_rand_absolute(self):
        bodyname = 'mid'
        attribute = 'pos'
        lower_bound_mult = np.array([0.5, 0.5, 0.5])
        upper_bound_mult = np.array([1.5, 1.5, 1.5])

        # ensure you have legit bounds for randomization
        assert (all(lower_bound_mult != 1))
        assert (all(upper_bound_mult != 1))
        assert (all(upper_bound_mult > lower_bound_mult))

        env = SwimmerEnv()
        body_id = env.sim.model._body_name2id[bodyname]  # get ID
        orig_val = np.array(env.sim.model.body_pos[body_id])
        print('Original', orig_val)  # get original attribute value

        variations = Variations()
        variations.randomize()\
            .at_xpath(f"//body[@name=\'{bodyname}\']")\
            .attribute(f"{attribute}")\
            .with_method(Method.ABSOLUTE)\
            .sampled_from(Distribution.UNIFORM)\
            .with_range(lower_bound_mult, upper_bound_mult)\
            .add()

        randomized_env = randomize(env, variations)
        randomized_vals = []

        for i in range(5):
            randomized_env.reset()
            body_id = randomized_env.wrapped_env.sim.model._body_name2id[
                bodyname]
            randomized_val = np.array(
                randomized_env.wrapped_env.sim.model.body_pos[body_id])
            randomized_vals.append(randomized_val)
            print('Randomised', randomized_val)  # see if attribute randomizes

            # check if within range
            assert (all(lower_bound_mult <= randomized_val)), \
                f"LowerBound {lower_bound_mult} not less that " \
                f"Randomised {randomized_val}"
            assert (all(randomized_val <= upper_bound_mult)), \
                f"UpperBound {upper_bound_mult} not greater than " \
                f"Randomised {randomized_val}"

            for j in range(5):
                randomized_env.step(randomized_env.action_space.sample())

        # check that you have actual variation
        randomized_vals = np.array(randomized_vals)
        print("Std dev of randomized values", np.std(randomized_vals))
        assert (np.std(randomized_vals) > 0), f'Std Dev of randomized values' \
            f' not > 0. Getting the exact same numbers?\n {randomized_vals}'

    def test_dynamics_rand_scaled(self):
        bodyname = 'mid'
        attribute = 'pos'
        lower_bound_mult = np.array([0.5, 0.5, 0.5])
        upper_bound_mult = np.array([1.5, 1.5, 1.5])

        # ensure you have legit bounds for randomization
        assert (all(lower_bound_mult != 1))
        assert (all(upper_bound_mult != 1))
        assert (all(upper_bound_mult > lower_bound_mult))

        env = SwimmerEnv()
        body_id = env.sim.model._body_name2id[bodyname]  # get ID
        orig_val = np.array(env.sim.model.body_pos[body_id])
        print('Original', orig_val)  # get original attribute value

        variations = Variations()
        variations.randomize()\
            .at_xpath(f"//body[@name=\'{bodyname}\']")\
            .attribute(f"{attribute}")\
            .with_method(Method.COEFFICIENT)\
            .sampled_from(Distribution.UNIFORM)\
            .with_range(lower_bound_mult, upper_bound_mult)\
            .add()

        randomized_env = randomize(env, variations)
        randomized_vals = []

        for i in range(5):
            randomized_env.reset()
            body_id = randomized_env.wrapped_env.sim.model._body_name2id[
                bodyname]
            randomized_val = np.array(
                randomized_env.wrapped_env.sim.model.body_pos[body_id])
            randomized_vals.append(randomized_val)
            print('Randomised', randomized_val)  # see if attribute randomizes

            # check if within range
            assert (all(orig_val * lower_bound_mult <= randomized_val)), \
                f"LowerBound {orig_val * lower_bound_mult} not < Randomised " \
                f"{randomized_val}"
            assert (all(randomized_val <= orig_val * upper_bound_mult)), \
                f"UpperBound {orig_val * upper_bound_mult} not > Randomised " \
                f"{randomized_val}"

            for j in range(5):
                randomized_env.step(randomized_env.action_space.sample())

        # check that you have actual variation
        randomized_vals = np.array(randomized_vals)
        print("Std dev of randomized values", np.std(randomized_vals))
        assert (np.std(randomized_vals) > 0), f'Std Dev of randomized values' \
            f' not > 0. Getting the exact same numbers?\n {randomized_vals}'
