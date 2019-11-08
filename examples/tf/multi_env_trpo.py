#!/usr/bin/env python3
"""This is an example to train multiple tasks with TRPO algorithm."""
from garage.envs import normalize
from garage.envs import PointEnv
from garage.envs.multi_env_wrapper import MultiEnvWrapper
from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy


def run_task(snapshot_config, *_):
    """Run task.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.

        _ (object): Ignored by this function.

    """
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env1 = TfEnv(normalize(PointEnv(goal=(-1., 0.))))
        env2 = TfEnv(normalize(PointEnv(goal=(1., 0.))))
        env = MultiEnvWrapper([env1, env2])

        policy = GaussianMLPPolicy(env_spec=env.spec)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=100,
                    discount=0.99,
                    gae_lambda=0.95,
                    lr_clip_range=0.2,
                    policy_ent_coeff=0.0)

        runner.setup(algo, env)
        runner.train(n_epochs=40, batch_size=2048, plot=False)


run_experiment(run_task, snapshot_mode='last', seed=1)
