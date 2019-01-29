#!/usr/bin/env python3
"""
This is an example to train a task with TRPO algorithm. It uses an LSTM-based
recurrent policy. To use a GRU-based recurrent policy, swap the commented
lines.

Here it runs CartPole-v1 environment with 100 iterations.

Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
from garage.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
import garage.tf.core.layers as L
from garage.tf.envs import TfEnv
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import CategoricalLSTMPolicy

env = TfEnv(env_name="CartPole-v1")

policy = CategoricalLSTMPolicy(
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
    n_itr=100,
    discount=0.99,
    max_kl_step=0.01,
    optimizer=ConjugateGradientOptimizer,
    optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)))
algo.train()
