from rllab.algos import TRPO
from rllab.baselines import LinearFeatureBaseline
from rllab.envs.mujoco import SwimmerEnv
from rllab.envs.mujoco.dynamics_randomization import Distribution
from rllab.envs.mujoco.dynamics_randomization import Method
from rllab.envs.mujoco.dynamics_randomization import randomize
from rllab.envs.mujoco.dynamics_randomization import Variations
from rllab.policies import GaussianMLPPolicy

variations = Variations()
variations.randomize() \
        .at_xpath(".//geom[@name='torso']") \
        .attribute("density") \
        .with_method(Method.COEFFICIENT) \
        .sampled_from(Distribution.UNIFORM) \
        .with_range(0.5, 1.5) \
        .add()

env = randomize(SwimmerEnv(), variations)

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=500,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
    # plot=True
)
algo.train()
