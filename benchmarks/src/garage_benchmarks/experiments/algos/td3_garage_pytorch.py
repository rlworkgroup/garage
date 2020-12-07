"""A regression test for automatic benchmarking garage-Pytorch-TD3."""
import torch
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.np.exploration_policies import AddGaussianNoise
from garage.np.policies import UniformRandomPolicy
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import prefer_gpu
from garage.torch.algos import TD3
from garage.torch.policies import DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

hyper_parameters = {
    'policy_lr': 1e-3,
    'qf_lr': 1e-3,
    'policy_hidden_sizes': [256, 256],
    'qf_hidden_sizes': [256, 256],
    'n_epochs': 250,
    'steps_per_epoch': 40,
    'batch_size': 100,
    'start_steps': 1000,
    'update_after': 1000,
    'grad_steps_per_env_step': 50,
    'discount': 0.99,
    'target_update_tau': 0.005,
    'replay_buffer_size': int(1e6),
    'sigma': 0.1,
    'policy_noise': 0.2,
    'policy_noise_clip': 0.5,
    'buffer_batch_size': 100,
    'min_buffer_size': int(1e4),
}


@wrap_experiment(snapshot_mode='last')
def td3_garage_pytorch(ctxt, env_id, seed):
    """Create garage TensorFlow TD3 model and training.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Localtrainer to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    trainer = Trainer(ctxt)

    num_timesteps = hyper_parameters['n_epochs'] * hyper_parameters[
        'steps_per_epoch'] * hyper_parameters['batch_size']
    env = normalize(GymEnv(env_id))

    policy = DeterministicMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=hyper_parameters['policy_hidden_sizes'],
        hidden_nonlinearity=F.relu,
        output_nonlinearity=torch.tanh)

    exploration_policy = AddGaussianNoise(env.spec,
                                          policy,
                                          total_timesteps=num_timesteps,
                                          max_sigma=hyper_parameters['sigma'],
                                          min_sigma=hyper_parameters['sigma'])

    uniform_random_policy = UniformRandomPolicy(env.spec)

    qf1 = ContinuousMLPQFunction(
        env_spec=env.spec,
        hidden_sizes=hyper_parameters['qf_hidden_sizes'],
        hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(
        env_spec=env.spec,
        hidden_sizes=hyper_parameters['qf_hidden_sizes'],
        hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(
        capacity_in_transitions=hyper_parameters['replay_buffer_size'])

    sampler = LocalSampler(agents=exploration_policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker)

    td3 = TD3(
        env_spec=env.spec,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        sampler=sampler,
        exploration_policy=exploration_policy,
        uniform_random_policy=uniform_random_policy,
        replay_buffer=replay_buffer,
        steps_per_epoch=hyper_parameters['steps_per_epoch'],
        policy_lr=hyper_parameters['policy_lr'],
        qf_lr=hyper_parameters['qf_lr'],
        target_update_tau=hyper_parameters['target_update_tau'],
        discount=hyper_parameters['discount'],
        grad_steps_per_env_step=hyper_parameters['grad_steps_per_env_step'],
        start_steps=hyper_parameters['start_steps'],
        min_buffer_size=hyper_parameters['min_buffer_size'],
        buffer_batch_size=hyper_parameters['buffer_batch_size'],
        policy_optimizer=torch.optim.Adam,
        qf_optimizer=torch.optim.Adam,
        policy_noise_clip=hyper_parameters['policy_noise_clip'],
        policy_noise=hyper_parameters['policy_noise'])

    prefer_gpu()
    td3.to()
    trainer.setup(td3, env)
    trainer.train(n_epochs=hyper_parameters['n_epochs'],
                  batch_size=hyper_parameters['batch_size'])
