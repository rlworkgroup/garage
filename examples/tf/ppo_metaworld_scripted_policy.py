#!/usr/bin/env python3
"""This is an example to train a task with PPO algorithm.

It creates Metaworld environmetns. And uses a PPO with 10M
steps.

"""

import click
import gym
import numpy as np
import tensorflow as tf

from metaworld.policies import *
from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS, ALL_V2_ENVIRONMENTS

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.policies import GaussianMLPPolicy
from garage.envs.wrappers import MWScriptedReward


_scripted_policies = {'lever-pull-v2' : SawyerLeverPullV2Policy,
                      'reach-v2' : SawyerReachV2Policy,
                      'pick-place-v2' : SawyerPickPlaceV2Policy,
                      'sweep-into-v1' : SawyerSweepIntoV1Policy
                     }

@click.command()
@click.option('--env-name', type=str)
@click.option('--seed', type=int, default=np.random.randint(0, 1000))
@wrap_experiment(name_parameters='all', snapshot_mode='gap', snapshot_gap=50)
def ppo_metaworld(ctxt=None, env_name=None, seed=1):
    """Train PPO with Metaworl environments.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    not_in_mw = "the env_name specified is not a metaworld environment"
    special = {'push-v1' : "push", 'reach-v1' : "reach", "pick-place-v1": "pick_place"}
    assert env_name in ALL_V2_ENVIRONMENTS or env_name in ALL_V1_ENVIRONMENTS, not_in_mw

    if env_name in ALL_V2_ENVIRONMENTS:
        env_cls = ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls()
    if env_name in special:
        env._set_task_inner(task_type=special[env_name])

    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.reset()
    env._freeze_rand_vec = True
    max_path_length = env.max_path_length

    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = GarageEnv(MWScriptedReward(env, scripted_policy=_scripted_policies[env_name]()))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            use_trust_region=True,
        )

        # NOTE: make sure when setting entropy_method to 'max', set
        # center_adv to False and turn off policy gradient. See
        # tf.algos.NPO for detailed documentation.
        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_episode_length=max_path_length,
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
        )

        runner.setup(algo, env)
        runner.train(n_epochs=int(10000000/(max_path_length*100)), batch_size=(max_path_length*100), plot=False)


ppo_metaworld()
