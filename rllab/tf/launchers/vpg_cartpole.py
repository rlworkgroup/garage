from rllab.baselines import LinearFeatureBaseline
from rllab.envs import normalize
from rllab.envs.box2d import CartpoleEnv
from rllab.misc import stub, run_experiment_lite
from rllab.tf.algos import VPG
from rllab.tf.envs import TfEnv
from rllab.tf.policies import GaussianMLPPolicy

env = TfEnv(normalize(CartpoleEnv()))

policy = GaussianMLPPolicy(
    name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = VPG(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=10000,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    optimizer_args=dict(tf_optimizer_args=dict(learning_rate=0.01, )))
algo.train()
