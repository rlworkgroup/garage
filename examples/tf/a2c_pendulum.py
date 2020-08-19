#!/usr/bin/env python3
"""Train A2C on InvertedDoublePendulum-v2 with 100 Iterations.

Results:
    AverageDiscountedReturn: 374
    RiseTime: itr 100

"""
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.tf.algos import A2C
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.policies import GaussianMLPPolicy


@wrap_experiment
def a2c_pendulum(ctxt=None, seed=1):
    """Train A2C with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
        :class:`~LocalRunner` to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = normalize(GymEnv('InvertedDoublePendulum-v2'))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
        )

        algo = A2C(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_episode_length=env.spec.max_episode_length,
            stop_entropy_gradient=True,
            center_adv=False,
            discount=0.99,
            optimizer_args=dict(learning_rate=1e-3),
        )

        runner.setup(algo, env)

        runner.train(n_epochs=500, batch_size=2000)


a2c_pendulum(seed=2)
