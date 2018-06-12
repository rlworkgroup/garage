from rllab.baselines import LinearFeatureBaseline
from rllab.envs import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.tf.algos import TRPO
import rllab.tf.core.layers as L
from rllab.envs.box2d import CartpoleEnv
from rllab.tf.envs import TfEnv
from rllab.tf.optimizers import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.tf.policies import GaussianGRUPolicy
from rllab.tf.policies import GaussianLSTMPolicy

env = TfEnv(normalize(CartpoleEnv()))

policy = GaussianLSTMPolicy(
    name="policy",
    env_spec=env.spec,
    lstm_layer_cls=L.TfBasicLSTMLayer,
    # gru_layer_cls=L.GRULayer,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=10,
    discount=0.99,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(
        hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)))
algo.train()
