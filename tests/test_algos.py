import os

from rllab.algos import CEM
from rllab.algos import CMAES
from rllab.algos import ERWR

os.environ['THEANO_FLAGS'] = 'device=cpu,mode=FAST_COMPILE,optimizer=None'

from rllab.algos import VPG
from rllab.algos import TNPG
from rllab.algos import PPO
from rllab.algos import TRPO
from rllab.algos import REPS
from rllab.algos import DDPG
from rllab.envs import GridWorldEnv
from rllab.envs.box2d import CartpoleEnv
from rllab.policies import CategoricalMLPPolicy
from rllab.policies import CategoricalGRUPolicy
from rllab.policies import GaussianMLPPolicy
from rllab.policies import GaussianGRUPolicy
from rllab.policies import DeterministicMLPPolicy
from rllab.q_functions import ContinuousMLPQFunction
from rllab.exploration_strategies import OUStrategy
from rllab.baselines import ZeroBaseline
from nose2 import tools
import numpy as np

common_batch_algo_args = dict(
    n_itr=1,
    batch_size=1000,
    max_path_length=100,
)

algo_args = {
    VPG:
    common_batch_algo_args,
    TNPG:
    dict(
        common_batch_algo_args,
        optimizer_args=dict(cg_iters=1, ),
    ),
    TRPO:
    dict(
        common_batch_algo_args,
        optimizer_args=dict(cg_iters=1, ),
    ),
    PPO:
    dict(
        common_batch_algo_args,
        optimizer_args=dict(max_penalty_itr=1, max_opt_itr=1),
    ),
    REPS:
    dict(
        common_batch_algo_args,
        max_opt_itr=1,
    ),
    DDPG:
    dict(
        n_epochs=1,
        epoch_length=100,
        batch_size=32,
        min_pool_size=50,
        replay_pool_size=1000,
        eval_samples=100,
    ),
    CEM:
    dict(
        n_itr=1,
        max_path_length=100,
        n_samples=5,
    ),
    CMAES:
    dict(
        n_itr=1,
        max_path_length=100,
        batch_size=1000,
    ),
    ERWR:
    common_batch_algo_args,
}

polopt_cases = []
for algo in [VPG, TNPG, PPO, TRPO, CEM, CMAES, ERWR, REPS]:
    polopt_cases.extend([
        (algo, GridWorldEnv, CategoricalMLPPolicy),
        (algo, CartpoleEnv, GaussianMLPPolicy),
        (algo, GridWorldEnv, CategoricalGRUPolicy),
        (algo, CartpoleEnv, GaussianGRUPolicy),
    ])


@tools.params(*polopt_cases)
def test_polopt_algo(algo_cls, env_cls, policy_cls):
    print("Testing %s, %s, %s" % (algo_cls.__name__, env_cls.__name__,
                                  policy_cls.__name__))
    env = env_cls()
    policy = policy_cls(env_spec=env.spec, )
    baseline = ZeroBaseline(env_spec=env.spec)
    algo = algo_cls(
        env=env,
        policy=policy,
        baseline=baseline,
        **(algo_args.get(algo_cls, dict())))
    algo.train()
    assert not np.any(np.isnan(policy.get_param_values()))


def test_ddpg():
    env = CartpoleEnv()
    policy = DeterministicMLPPolicy(env.spec)
    qf = ContinuousMLPQFunction(env.spec)
    es = OUStrategy(env.spec)
    algo = DDPG(
        env=env,
        policy=policy,
        qf=qf,
        es=es,
        n_epochs=1,
        epoch_length=100,
        batch_size=32,
        min_pool_size=50,
        replay_pool_size=1000,
        eval_samples=100,
    )
    algo.train()
