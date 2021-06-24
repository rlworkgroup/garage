#!/usr/bin/env python3
"""This is an example to train TRPO on MT1 environment."""
# pylint: disable=no-value-for-parameter
import click
import metaworld
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import MetaWorldTaskSampler
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.tf.algos import TRPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer


@click.command()
@click.option('--seed', default=1)
@click.option('--entropy', default=5e-3)
@click.option('--env-name')
@wrap_experiment(snapshot_mode='none', name_parameters='passed')
def mttrpo_metaworld_MT1(ctxt, seed, entropy, env_name):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        entropy (float): Coefficient to weigh the entropy reward term by
            when using the max entropy reward.
        env_name (str): Name of Meta-World environment to initialize
            this experiment with.

    """
    n_tasks = 50
    set_seed(seed)
    MT1 = metaworld.MT1(env_name)
    train_task_sampler = MetaWorldTaskSampler(MT1,
                                              'train',
                                              lambda env, _: normalize(env),
                                              add_env_onehot=True)
    assert n_tasks % 10 == 0
    assert n_tasks <= 500
    envs = [env_up() for env_up in train_task_sampler.sample(n_tasks)]
    env = envs[0]
    with TFTrainer(snapshot_config=ctxt) as trainer:
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(512, 512),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
            std_share_network=True,
            min_std=0.5,
            max_std=1.5,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            hidden_sizes=(512, 512),
            use_trust_region=True,
        )

        sampler = RaySampler(agents=policy,
                             envs=envs,
                             max_episode_length=env.spec.max_episode_length,
                             n_workers=50,
                             is_tf_worker=True)

        algo = TRPO(env_spec=env.spec,
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
                    use_neg_logli_entropy=True,
                    multitask=True)

        trainer.setup(algo, env)
        trainer.train(n_epochs=int(100000000 / (500 * 100)),
                      batch_size=2,
                      plot=False)


mttrpo_metaworld_MT1()
