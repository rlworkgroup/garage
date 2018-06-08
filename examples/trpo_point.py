from examples.point_env import PointEnv
from rllab.algos import TRPO
from rllab.baselines import LinearFeatureBaseline
from rllab.envs import normalize
from rllab.envs.gym_util.env_util import spec
from rllab.policies import GaussianMLPPolicy

env = normalize(PointEnv())
<<<<<<< HEAD
policy = GaussianMLPPolicy(env_spec=env.spec, )
baseline = LinearFeatureBaseline(env_spec=env.spec)
=======
policy = GaussianMLPPolicy(env_spec=spec(env), )
baseline = LinearFeatureBaseline(env_spec=spec(env))
>>>>>>> Refactored rllab.Env to gym.Env
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
)
algo.train()
