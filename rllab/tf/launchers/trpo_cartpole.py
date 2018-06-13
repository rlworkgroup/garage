from rllab.baselines import LinearFeatureBaseline
from rllab.envs.box2d import CartpoleEnv
from rllab.envs import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.tf.algos import TRPO
from rllab.tf.envs import TfEnv
from rllab.tf.optimizers import ConjugateGradientOptimizer
from rllab.tf.optimizers import FiniteDifferenceHvp
from rllab.tf.policies import GaussianMLPPolicy

env = TfEnv(normalize(CartpoleEnv()))

policy = GaussianMLPPolicy(
    name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
    plot=True)
algo.train()
