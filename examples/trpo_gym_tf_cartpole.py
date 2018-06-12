from rllab.baselines import LinearFeatureBaseline
from rllab.envs import GymEnv
from rllab.envs import normalize
from rllab.misc import run_experiment_lite, stub
from rllab.tf.algos import TRPO
from rllab.tf.envs import TfEnv
from rllab.tf.policies import CategoricalMLPPolicy

stub(globals())

# Need to wrap in a tf environment and force_reset to true
# see https://github.com/openai/rllab/issues/87#issuecomment-282519288
env = TfEnv(normalize(GymEnv("CartPole-v0", force_reset=True)))

policy = CategoricalMLPPolicy(
    name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=200,
    n_itr=120,
    discount=0.99,
    step_size=0.01,
)

run_experiment_lite(algo.train(), n_parallel=1, snapshot_mode="last", seed=1)
