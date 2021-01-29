#!/usr/bin/env python3
"""This is an example to train TRPO on MT10 environment."""
# pylint: disable=no-value-for-parameter
import click
import metaworld
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import normalize
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment import MetaWorldTaskSampler
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.tf.algos import TRPO
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.baselines import GaussianMLPBaseline
from garage.trainer import TFTrainer
from garage.tf.optimizers import FirstOrderOptimizer


@click.command()
@click.option('--seed', default=1)
@click.option('--entropy', default=5e-3)
@wrap_experiment(snapshot_mode='none', name_parameters='passed')
def mttrpo_metaworld_mt10(ctxt, seed, entropy):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        batch_size (int): Number of environment steps in one batch.
        n_tasks (int): Number of tasks to use. Should be a multiple of 10.

    """
    n_tasks = 10
    set_seed(seed)
    mt10 = metaworld.MT10()
    train_task_sampler = MetaWorldTaskSampler(mt10,
                                              'train',
                                              lambda env, _: normalize(env),
                                              add_env_onehot=True)
    assert n_tasks % 10 == 0
    assert n_tasks <= 500
    envs = [env_up() for env_up in train_task_sampler.sample(n_tasks)]
    env = MultiEnvWrapper(envs,
                          sample_strategy=round_robin_strategy,
                          mode='vanilla')
    with TFTrainer(snapshot_config=ctxt) as trainer:
        policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(256, 256),
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=None,
                std_share_network=True,
                min_std=0.5,
                max_std=1.5,
            )

        baseline = GaussianMLPBaseline(
                env_spec=env.spec,
                hidden_sizes=(128, 128),
                use_trust_region=True,
            )

        sampler = RaySampler(agents=policy,
                            envs=env,
                            max_episode_length=env.spec.max_episode_length,
                            n_workers=10,
                            is_tf_worker=True)

        algo = TRPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                discount=0.99,
                gae_lambda=0.95,
                lr_clip_range=0.2,
                stop_entropy_gradient=True,
                entropy_method='max',
                policy_ent_coeff=entropy,
                center_adv=False,
                use_softplus_entropy=False,
                sampler=sampler,
                use_neg_logli_entropy=True
            )

        trainer.setup(algo, env)
        trainer.train(n_epochs=int(20000000 / (500 * 100)),
                    batch_size=(500 * 100),
                    plot=False)


mttrpo_metaworld_mt10()