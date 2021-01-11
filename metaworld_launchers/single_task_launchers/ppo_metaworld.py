#!/usr/bin/env python3
"""This is an example to train a task with PPO algorithm.

It creates Metaworld environmetns. And uses a PPO with 10M
steps.

"""

import pickle

import click
import gym
from metaworld.envs.mujoco.env_dict import (ALL_V1_ENVIRONMENTS,
                                            ALL_V2_ENVIRONMENTS)
import numpy as np
import torch

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.trainer import Trainer
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler, RaySampler
from garage.torch.algos import PPO
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.optimizers import OptimizerWrapper

# import pip
# package = f'metaworld @ https://git@api.github.com/repos/rlworkgroup/metaworld/tarball/new-reward-functions'
# pip.main(['install', '--upgrade', package])

@click.command()
@click.option('--env-name', type=str, default='push-v2')
@click.option('--seed', type=int, default=np.random.randint(0, 1000))
@click.option('--entropy', type=float, default=0.01)
@click.option('--use_softplus_entropy', type=bool, default=False)
@click.option('--extra_tags', type=str, default='none')
@wrap_experiment(name_parameters='all', snapshot_mode='gap', snapshot_gap=25)
def ppo_metaworld(
    ctxt=None,
    env_name=None,
    tag='',
    extra_tags='',
    entropy=0.01,
    stop_entropy_gradient=True,
    use_softplus_entropy=False,
    seed=1,
):
    """Train PPO with Metaworld environments.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    special = {
        'push-v1': 'push',
        'reach-v1': 'reach',
        'pick-place-v1': 'pick_place'
    }
    not_in_mw = 'the env_name specified is not a metaworld environment'
    assert env_name in ALL_V2_ENVIRONMENTS or env_name in ALL_V1_ENVIRONMENTS, not_in_mw

    if env_name in ALL_V2_ENVIRONMENTS:
        env_cls = ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.reset()
    env._freeze_rand_vec = True
    max_path_length = env.max_path_length
    env = normalize(GymEnv(env, max_episode_length=max_path_length), normalize_reward=True)
    trainer = Trainer(snapshot_config=ctxt)
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    baseline = GaussianMLPValueFunction(
        env_spec=env.spec,
        hidden_sizes=(128, 128),
    )

    sampler= RaySampler(agents=policy,
                        envs=env,
                        max_episode_length=env.spec.max_episode_length,)
    # sampler = LocalSampler(agents=policy,
    #                        envs=env,
    #                        max_episode_length=env.spec.max_episode_length,
    #                        is_tf_worker=True)

    # NOTE: make sure when setting entropy_method to 'max', set
    # center_adv to False and turn off policy gradient. See
    # tf.algos.NPO for detailed documentation.
    policy_optimizer = OptimizerWrapper(
                        (torch.optim.Adam, dict(lr=5e-4)),
                        policy,
                        max_optimization_epochs=10,
                        minibatch_size=64)
    vf_optimizer = OptimizerWrapper(
                        (torch.optim.Adam, dict(lr=5e-4)),
                        baseline,
                        max_optimization_epochs=10,
                        minibatch_size=64)
    algo = PPO(
        env_spec=env.spec,
        policy=policy,
        value_function=baseline,
        discount=0.99,
        gae_lambda=0.95,
        stop_entropy_gradient=stop_entropy_gradient,
        entropy_method='max',
        policy_ent_coeff=entropy,
        center_adv=False,
        use_softplus_entropy=use_softplus_entropy,
        sampler=sampler,
        policy_optimizer=policy_optimizer,
        vf_optimizer=vf_optimizer
    )

    trainer.setup(algo, env)
    trainer.train(n_epochs=int(20000000 / (max_path_length * 100)),
                    batch_size=(max_path_length * 100),
                    plot=False)


ppo_metaworld()
