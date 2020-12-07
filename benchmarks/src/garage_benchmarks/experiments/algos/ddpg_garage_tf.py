"""A regression test for automatic benchmarking garage-TensorFlow-DDPG."""
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.tf.algos import DDPG
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.trainer import TFTrainer

hyper_parameters = {
    'policy_lr': 1e-4,
    'qf_lr': 1e-3,
    'policy_hidden_sizes': [64, 64],
    'qf_hidden_sizes': [64, 64],
    'n_epochs': 500,
    'steps_per_epoch': 20,
    'n_exploration_steps': 100,
    'n_train_steps': 50,
    'discount': 0.9,
    'tau': 1e-2,
    'replay_buffer_size': int(1e6),
    'sigma': 0.2
}


@wrap_experiment
def ddpg_garage_tf(ctxt, env_id, seed):
    """Create garage TensorFlow DDPG model and training.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    with TFTrainer(ctxt) as trainer:
        env = normalize(GymEnv(env_id))

        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['policy_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)

        exploration_policy = AddOrnsteinUhlenbeckNoise(
            env.spec, policy, sigma=hyper_parameters['sigma'])

        qf = ContinuousMLPQFunction(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['qf_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu)

        replay_buffer = PathBuffer(
            capacity_in_transitions=hyper_parameters['replay_buffer_size'])

        sampler = LocalSampler(agents=exploration_policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True,
                               worker_class=FragmentWorker)

        algo = DDPG(env_spec=env.spec,
                    policy=policy,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    sampler=sampler,
                    steps_per_epoch=hyper_parameters['steps_per_epoch'],
                    policy_lr=hyper_parameters['policy_lr'],
                    qf_lr=hyper_parameters['qf_lr'],
                    target_update_tau=hyper_parameters['tau'],
                    n_train_steps=hyper_parameters['n_train_steps'],
                    discount=hyper_parameters['discount'],
                    min_buffer_size=int(1e4),
                    exploration_policy=exploration_policy,
                    policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                    qf_optimizer=tf.compat.v1.train.AdamOptimizer)

        trainer.setup(algo, env)
        trainer.train(n_epochs=hyper_parameters['n_epochs'],
                      batch_size=hyper_parameters['n_exploration_steps'])
