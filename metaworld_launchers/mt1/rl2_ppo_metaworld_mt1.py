#!/usr/bin/env python3
"""Example script to run RL2 in ML1."""
# pylint: disable=no-value-for-parameter
# yapf: disable
import click
import metaworld
import copy

from numpy.core.records import _deprecate_shape_0_as_None

from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage.experiment import (MetaEvaluator, MetaWorldTaskSampler,
                               SetTaskSampler)
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.optimizers import FirstOrderOptimizer
from garage.sampler import RaySampler
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env, RL2Worker
from garage.tf.policies import GaussianGRUPolicy
from garage.trainer import TFTrainer

# yapf: enable


@click.command()
@click.option('--env-name', type=str)
@click.option('--seed', default=1)
@click.option('--meta_batch_size', default=10)
@click.option('--n_epochs', default=2000)
@click.option('--episode_per_task', default=10)
@click.option('--entropy_coefficient', default=5e-6)
@wrap_experiment(snapshot_mode='gap', snapshot_gap=100, name_parameters='all')
def rl2_ppo_metaworld_mt1(ctxt, env_name, seed, meta_batch_size, n_epochs,
                               episode_per_task,
                               entropy_coefficient,
                               extra_tags="use_neg_logli_entropy=True_std_share_network=True_stddev_clipping=SoftClip"):
    """Train RL2 PPO with MT1 environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.
        seed (int): Used to seed the random number generator to produce
            determinism.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.

    """
    set_seed(seed)
    ml1 = metaworld.MT1(env_name)

    task_sampler = MetaWorldTaskSampler(ml1, 'train',
                                        lambda env, _: RL2Env(normalize(env, normalize_reward=True)))
    env = task_sampler.sample(1)[0]()
    test_task_sampler = copy.deepcopy(task_sampler)
    env_spec = env.spec

    with TFTrainer(snapshot_config=ctxt) as trainer:
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=256,
                                   env_spec=env_spec,
                                   state_include_action=False,
                                   std_share_network=True,
                                   init_std=0.5,
                                   max_std=1.2,
                                   min_std=0.5)

        meta_evaluator = MetaEvaluator(test_task_sampler=test_task_sampler)

        baseline = baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            hidden_sizes=(128, 128),
            use_trust_region=False,
            optimizer=FirstOrderOptimizer,
            optimizer_args=dict(
                batch_size=32,
                max_optimization_epochs=10,
                learning_rate=3e-4,
            ),
        )

        envs = task_sampler.sample(meta_batch_size)
        sampler = RaySampler(
            agents=policy,
            envs=envs,
            max_episode_length=env_spec.max_episode_length,
            is_tf_worker=True,
            n_workers=meta_batch_size,
            worker_class=RL2Worker,
            worker_args=dict(n_episodes_per_trial=episode_per_task))

        algo = RL2PPO(meta_batch_size=meta_batch_size,
                      task_sampler=task_sampler,
                      env_spec=env_spec,
                      policy=policy,
                      baseline=baseline,
                      sampler=sampler,
                      discount=0.99,
                      gae_lambda=1,
                      lr_clip_range=0.2,
                      optimizer_args=dict(batch_size=32,
                                          max_optimization_epochs=10,
                                          learning_rate=5e-4),
                      stop_entropy_gradient=True,
                      entropy_method='max',
                      policy_ent_coeff=entropy_coefficient,
                      center_adv=False,
                      meta_evaluator=meta_evaluator,
                      episodes_per_trial=episode_per_task,
                      use_neg_logli_entropy=True)

        trainer.setup(algo, envs)

        trainer.train(n_epochs=n_epochs,
                      batch_size=episode_per_task *
                      env_spec.max_episode_length * meta_batch_size)


rl2_ppo_metaworld_mt1()
