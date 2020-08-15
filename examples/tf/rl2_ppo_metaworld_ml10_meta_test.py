#!/usr/bin/env python3
"""Example script to run RL2 in ML10 with meta-test."""
# pylint: disable=no-value-for-parameter
import click
import metaworld.benchmarks as mwb

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment import LocalTFRunner, task_sampler
from garage.experiment.deterministic import set_seed
from garage.experiment.meta_evaluator import MetaEvaluator
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env, RL2Worker
from garage.tf.policies import GaussianGRUPolicy


@click.command()
@click.option('--seed', default=1)
@click.option('--max_episode_length', default=150)
@click.option('--meta_batch_size', default=10)
@click.option('--n_epochs', default=10)
@click.option('--episode_per_task', default=10)
@wrap_experiment
def rl2_ppo_metaworld_ml10_meta_test(ctxt, seed, max_episode_length,
                                     meta_batch_size, n_epochs,
                                     episode_per_task):
    """Train PPO with ML10 environment with meta-test.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~LocalRunner` to create the :class:`~Snapshotter`.
        seed (int): Used to seed the random number generator to produce
            determinism.
        max_episode_length (int): Maximum length of a single episode.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        ml10_train_envs = [
            RL2Env(GymEnv(mwb.ML10.from_task(task_name)))
            for task_name in mwb.ML10.get_train_tasks().all_task_names
        ]
        tasks = task_sampler.EnvPoolSampler(ml10_train_envs)
        tasks.grow_pool(meta_batch_size)

        ml10_test_envs = [
            RL2Env(GymEnv(mwb.ML10.from_task(task_name)))
            for task_name in mwb.ML10.get_test_tasks().all_task_names
        ]
        test_tasks = task_sampler.EnvPoolSampler(ml10_test_envs)

        env_spec = ml10_train_envs[0].spec
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=64,
                                   env_spec=env_spec,
                                   state_include_action=False)

        baseline = LinearFeatureBaseline(env_spec=env_spec)

        meta_evaluator = MetaEvaluator(test_task_sampler=test_tasks,
                                       n_exploration_eps=10,
                                       n_test_episodes=10,
                                       max_episode_length=max_episode_length,
                                       n_test_tasks=5)

        algo = RL2PPO(rl2_max_episode_length=max_episode_length,
                      meta_batch_size=meta_batch_size,
                      task_sampler=tasks,
                      env_spec=env_spec,
                      policy=policy,
                      baseline=baseline,
                      discount=0.99,
                      gae_lambda=0.95,
                      lr_clip_range=0.2,
                      optimizer_args=dict(
                          batch_size=32,
                          max_episode_length=10,
                      ),
                      stop_entropy_gradient=True,
                      entropy_method='max',
                      policy_ent_coeff=0.02,
                      center_adv=False,
                      max_episode_length=max_episode_length * episode_per_task,
                      meta_evaluator=meta_evaluator,
                      n_epochs_per_eval=10)

        runner.setup(algo,
                     tasks.sample(meta_batch_size),
                     sampler_cls=LocalSampler,
                     n_workers=meta_batch_size,
                     worker_class=RL2Worker,
                     worker_args=dict(n_episodes_per_trial=episode_per_task))

        runner.train(n_epochs=n_epochs,
                     batch_size=episode_per_task * max_episode_length *
                     meta_batch_size)


rl2_ppo_metaworld_ml10_meta_test()
