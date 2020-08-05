#!/usr/bin/env python3
"""An example to train a task with TRPO algorithm."""
from garage import wrap_experiment
from garage.envs import GymEnv
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
        env = GymEnv('CartPole-v0')

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_episode_length=200,
            discount=0.99,
            max_kl_step=0.01,
        )

        runner.setup(algo, env)
        runner.train(n_epochs=120, batch_size=4000)


trpo_gym_tf_cartpole()
