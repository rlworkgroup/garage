#!/usr/bin/env python3
"""Example script to run RL2 in ML10."""
# pylint: disable=no-value-for-parameter
import click
import metaworld.benchmarks as mwb

from garage import wrap_experiment
from garage.experiment import task_sampler
from garage.experiment.deterministic import set_seed
from garage.experiment.meta_evaluator import MetaEvaluator
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler, RaySampler
from garage.tf.algos import RL2TRPO
from garage.tf.algos.rl2 import RL2Env
from garage.tf.algos.rl2 import RL2Worker
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import GaussianGRUPolicy


@click.command()
@click.option('--seed', default=1)
@click.option('--max_path_length', default=150)
@click.option('--meta_batch_size', default=40)
@click.option('--n_epochs', default=1500)
@click.option('--episode_per_task', default=10)
@wrap_experiment
def rl2_trpo_metaworld_ml10(ctxt, seed, max_path_length, meta_batch_size,
                           n_epochs, episode_per_task):
    """Train PPO with ML10 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        max_path_length (int): Maximum length of a single rollout.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        ml10_train_envs = [
            RL2Env(mwb.ML10.from_task(task_name))
            for task_name in mwb.ML10.get_train_tasks().all_task_names
        ]
        tasks = task_sampler.EnvPoolSampler(ml10_train_envs)
        tasks.grow_pool(meta_batch_size)

        ml10_test_envs = [
            RL2Env(mwb.ML10.from_task(task_name))
            for task_name in mwb.ML10.get_test_tasks().all_task_names
        ]
        test_tasks = task_sampler.EnvPoolSampler(ml10_test_envs)

        env_spec = ml10_train_envs[0].spec
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=[200, 200],
                                   env_spec=env_spec,
                                   state_include_action=False)

        baseline = LinearFeatureBaseline(env_spec=env_spec)

        meta_evaluator = MetaEvaluator(test_task_sampler=test_tasks,
                                       n_exploration_traj=10,
                                       n_test_rollouts=10,
                                       max_path_length=max_path_length,
                                       n_test_tasks=5)

        algo = RL2TRPO(rl2_max_path_length=max_path_length,
                       meta_batch_size=meta_batch_size,
                       task_sampler=tasks,
                       env_spec=env_spec,
                       policy=policy,
                       baseline=baseline,
                       max_path_length=max_path_length * episode_per_task,
                       discount=0.99,
                       max_kl_step=0.01,
                       lr_clip_range=0.2,
                       optimizer=ConjugateGradientOptimizer,
                       optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(
                           base_eps=1e-5)),
                       meta_evaluator=meta_evaluator,
                       n_epochs_per_eval=10)

        runner.setup(algo,
                     tasks.sample(meta_batch_size),
                     sampler_cls=LocalSampler,
                     n_workers=meta_batch_size,
                     worker_class=RL2Worker,
                     worker_args=dict(n_paths_per_trial=episode_per_task))

        runner.train(n_epochs=n_epochs,
                     batch_size=episode_per_task * max_path_length *
                     meta_batch_size)


rl2_trpo_metaworld_ml10()
