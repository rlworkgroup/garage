from rllab.envs.mujoco import SwimmerEnv
from rllab.envs.mujoco.randomization import Distribution
from rllab.envs.mujoco.randomization import Method
from rllab.envs.mujoco.randomization import randomize
from rllab.envs.mujoco.randomization import Variations

variations = Variations()
variations.randomize() \
        .at_xpath(".//geom[@name='torso']") \
        .attribute("density") \
        .with_method(Method.COEFFICIENT) \
        .sampled_from(Distribution.UNIFORM) \
        .with_range(0.5, 1.5) \
        .add()

env = randomize(SwimmerEnv(), variations)

for i in range(1000):
    env.reset()
    for j in range(1000):
        env.step(env.action_space.sample())
