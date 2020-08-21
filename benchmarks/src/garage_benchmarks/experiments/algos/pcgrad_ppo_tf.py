#!/usr/bin/env python3
"""This is an example to train multiple tasks with PCGrad-PPO algorithm."""
from metaworld.benchmarks import MT10
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.tf.algos import PPO
from garage.tf.algos.pcgrad import PCGradWorker
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.optimizers import PCGradOptimizer, FirstOrderOptimizer
from garage.tf.policies import GaussianMLPPolicy


@wrap_experiment
def pcgrad_ppo_tf(ctxt, env_id, seed):
    """Train PPO on MT10 environment with PCGrad optimizer.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(ctxt) as runner:
        tasks = MT10.get_train_tasks().all_task_names
        envs = [normalize(GymEnv(MT10.from_task(task))) for task in tasks]
        env = MultiEnvWrapper(envs,
                              sample_strategy=round_robin_strategy,
                              mode='vanilla')

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            use_trust_region=False,
            optimizer=FirstOrderOptimizer,
            optimizer_args=dict(
                batch_size=32,
                max_episode_length=10,
                learning_rate=3e-4,
            ),
        )

        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   max_episode_length=100,
                   discount=0.99,
                   gae_lambda=0.95,
                   center_adv=True,
                   lr_clip_range=0.2,
                   loss_group_by_task_id=True,
                   num_tasks=len(envs),
                   optimizer=PCGradOptimizer,
                   optimizer_args=dict(
                       batch_size=32,
                       max_episode_length=10,
                       learning_rate=3e-4,
                   ))

        runner.setup(algo,
                     env,
                     worker_class=PCGradWorker,
                     worker_args=dict(num_tasks=len(envs)))
        runner.train(n_epochs=800, batch_size=2048, plot=False)
