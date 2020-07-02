#!/usr/bin/env python3
"""An example to train a task with TRPO algorithm."""
import gym
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.policies import CategoricalMLPPolicy


@wrap_experiment
def trpo_gym_tf_cartpole(ctxt=None, seed=1):
    """Train TRPO with CartPole-v0 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = GarageEnv(gym.make('CartPole-v1'))

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=200,
            discount=0.99,
            max_kl_step=0.01,
        )

        runner.setup(algo, env)
        runner.train(n_epochs=10, batch_size=10000, plot=False)


@wrap_experiment
def pre_trained_trpo_cartpole(
        ctxt=None,
        snapshot_dir='data/local/experiment/trpo_gym_tf_cartpole',
        seed=1):
    """Use pre-trained TRPO and reusume experiment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        snapshot_dir (path): directory to snapshot
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        runner.restore(snapshot_dir)
        runner.resume(n_epochs=30, batch_size=8000)


if __name__ == '__main__':
    # To train a new TRPO
    trpo_gym_tf_cartpole()

    # Clear tensorflow graph
    tf.compat.v1.reset_default_graph()

    # To use a pretrained TRPO
    # snapshot_dir = 'data/local/experiment/trpo_gym_tf_cartpole'
    pre_trained_trpo_cartpole()
