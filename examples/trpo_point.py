from examples.point_env import PointEnv
from rllab.algos import TRPO
from rllab.baselines import LinearFeatureBaseline
from rllab.envs import normalize
from rllab.policies import GaussianMLPPolicy

env = normalize(PointEnv())
policy = GaussianMLPPolicy(env_spec=env.spec, )
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
)
algo.train()
