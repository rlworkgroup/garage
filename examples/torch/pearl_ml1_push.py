"""PEARL ML1 example."""

import os

import akro
from metaworld.benchmarks import ML1
import numpy as np

from garage.envs import EnvSpec, GarageEnv, normalize
from garage.experiment import LocalRunner, SnapshotConfig, wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.experiment.meta_evaluator import MetaEvaluator
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import LocalSampler
from garage.torch.algos import PEARL
from garage.torch.algos.pearl import PEARLWorker
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import (ContextConditionedPolicy,
                                   TanhGaussianMLPPolicy)
from garage.torch.q_functions import ContinuousMLPQFunction
import garage.torch.utils as tu

algo_params = dict(
    num_epochs=1000,
    num_train_tasks=50,
    num_test_tasks=10,
    latent_size=7,
    encoder_hidden_sizes=[200, 200, 200],
    net_size=300,
    meta_batch_size=16,
    num_steps_per_epoch=4000,
    num_initial_steps=4000,
    num_tasks_sample=15,
    num_steps_prior=750,
    num_extra_rl_steps_posterior=750,
    num_steps_per_eval=450,
    batch_size=256,
    embedding_batch_size=64,
    embedding_mini_batch_size=64,
    max_path_length=150,
    reward_scale=10.,
    use_information_bottleneck=True,
    use_next_obs_in_context=False,
    use_gpu=True,
)


# pylint: disable=unused-argument
@wrap_experiment
def torch_pearl_ml1_push(ctxt=None, seed=1, **params):
    """Train PEARL with ML1 environments.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        params (dict): Parameters for PEARL.

    """
    set_seed(seed)
    # create multi-task environment and sample tasks
    env_sampler = SetTaskSampler(lambda: GarageEnv(
        normalize(ML1.get_train_tasks('push-v1'))))
    env = env_sampler.sample(params['num_train_tasks'])

    test_env_sampler = SetTaskSampler(lambda: GarageEnv(
        normalize(ML1.get_test_tasks('push-v1'))))
    snapshot_config = SnapshotConfig(snapshot_dir=os.path.join(
        os.getcwd(), 'data/local/experiment'),
                                     snapshot_mode='last',
                                     snapshot_gap=1)
    runner = LocalRunner(snapshot_config)
    # instantiate networks
    net_size = params['net_size']
    obs_dim = max(
        int(np.prod(env[i]().observation_space.shape))
        for i in range(params['num_train_tasks']))
    action_dim = int(np.prod(env[0]().action_space.shape))

    space_a = akro.Box(low=-1,
                       high=1,
                       shape=(obs_dim + params['latent_size'], ),
                       dtype=np.float32)
    space_b = akro.Box(low=-1, high=1, shape=(action_dim, ), dtype=np.float32)
    augmented_env = EnvSpec(space_a, space_b)

    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size])

    obs_space = akro.Box(low=-1, high=1, shape=(obs_dim, ), dtype=np.float32)
    action_space = akro.Box(low=-1,
                            high=1,
                            shape=(params['latent_size'], ),
                            dtype=np.float32)
    vf_env = EnvSpec(obs_space, action_space)

    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

    pearl = PEARL(
        env=env,
        policy_class=ContextConditionedPolicy,
        encoder_class=MLPEncoder,
        inner_policy=inner_policy,
        qf=qf,
        vf=vf,
        num_train_tasks=params['num_train_tasks'],
        num_test_tasks=params['num_test_tasks'],
        latent_dim=params['latent_size'],
        encoder_hidden_sizes=params['encoder_hidden_sizes'],
        meta_batch_size=params['meta_batch_size'],
        num_steps_per_epoch=params['num_steps_per_epoch'],
        num_initial_steps=params['num_initial_steps'],
        num_tasks_sample=params['num_tasks_sample'],
        num_steps_prior=params['num_steps_prior'],
        num_extra_rl_steps_posterior=params['num_extra_rl_steps_posterior'],
        num_steps_per_eval=params['num_steps_per_eval'],
        batch_size=params['batch_size'],
        embedding_batch_size=params['embedding_batch_size'],
        embedding_mini_batch_size=params['embedding_mini_batch_size'],
        max_path_length=params['max_path_length'],
        reward_scale=params['reward_scale'],
    )

    tu.set_gpu_mode(params['use_gpu'], gpu_id=0)
    if params['use_gpu']:
        pearl.to()

    runner.setup(algo=pearl,
                 env=env[0](),
                 sampler_cls=LocalSampler,
                 sampler_args=dict(max_path_length=params['max_path_length']),
                 n_workers=1,
                 worker_class=PEARLWorker)

    worker_args = dict(deterministic=True, accum_context=True)
    meta_evaluator = MetaEvaluator(runner,
                                   test_task_sampler=test_env_sampler,
                                   max_path_length=params['max_path_length'],
                                   worker_class=PEARLWorker,
                                   worker_args=worker_args,
                                   n_test_tasks=params['num_test_tasks'])
    pearl.evaluator = meta_evaluator
    runner.train(n_epochs=params['num_epochs'],
                 batch_size=params['batch_size'])


torch_pearl_ml1_push(**algo_params)
