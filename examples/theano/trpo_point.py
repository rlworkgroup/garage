from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.point_env import PointEnv
from garage.theano.algos import TRPO
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy

env = TheanoEnv(normalize(PointEnv()))
policy = GaussianMLPPolicy(env_spec=env.spec, )
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
)
algo.train()
