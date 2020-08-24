"""A regression test for automatic benchmarking garage-Pytorch-TD3."""
import gym
import torch
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic, LocalTFRunner
from garage.np.exploration_policies import AddGaussianNoise
from garage.replay_buffer import PathBuffer
from garage.torch.algos import TD3
from garage.torch.policies import DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction

hyper_parameters = {
    'policy_lr': 1e-3,
    'qf_lr': 1e-3,
    'policy_hidden_sizes': [256, 256],
    'qf_hidden_sizes': [256, 256],
    'n_epochs': 250,
    'steps_per_epoch': 120,
    'batch_size': 100,
    'start_steps': 1000,
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


@wrap_experiment(snapshot_mode='none')
def td3_garage_pytorch(ctxt, env_id, seed):
    """Create garage TensorFlow TD3 model and training.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    with LocalTFRunner(ctxt) as runner:
        env = normalize(GymEnv(env_id))

        policy = DeterministicMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['policy_hidden_sizes'],
            hidden_nonlinearity=F.relu,
            output_nonlinearity=torch.tanh)

        exploration_policy = AddGaussianNoise(
            env.spec,
            policy,
            max_sigma=hyper_parameters['sigma'],
            min_sigma=hyper_parameters['sigma'])

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

        td3 = TD3(env_spec=env.spec,
                  policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  exploration_policy=exploration_policy,
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

        td3.to()
        runner.setup(td3, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'],
                     batch_size=hyper_parameters['batch_size'])
