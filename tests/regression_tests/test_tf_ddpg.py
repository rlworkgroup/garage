import argparse

from baselines import logger
from baselines.common.misc_util import boolean_flag
from baselines.ddpg.main import run
import gym
import tensorflow as tf
from mpi4py import MPI

from garage.misc.instrument import run_experiment
from garage.tf.algos import DDPG
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction


def test_ddpg():
    env = gym.make('Pendulum-v0')

    def run_task(*_):
        action_noise = OUStrategy(env, sigma=0.2)

        actor_net = ContinuousMLPPolicy(
            env_spec=env,
            name="Actor",
            hidden_sizes=[64, 64],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)

        critic_net = ContinuousMLPQFunction(
            env_spec=env,
            name="Critic",
            hidden_sizes=[64, 64],
            hidden_nonlinearity=tf.nn.relu,
        )

        ddpg = DDPG(
            env,
            actor=actor_net,
            critic=critic_net,
            plot=False,
            n_epochs=500,
            n_epoch_cycles=20,
            n_rollout_steps=100,
            n_train_steps=50,
            exploration_strategy=action_noise,
            actor_optimizer=tf.train.AdamOptimizer,
            critic_optimizer=tf.train.AdamOptimizer)

        ddpg.train()

    run_experiment(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        plot=False,
    )

    def parse_args():
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--env-id', type=str, default='Pendulum-v0')
        boolean_flag(parser, 'render-eval', default=False)
        boolean_flag(parser, 'layer-norm', default=False)
        boolean_flag(parser, 'render', default=False)
        boolean_flag(parser, 'normalize-returns', default=False)
        boolean_flag(parser, 'normalize-observations', default=False)
        parser.add_argument('--seed', help='RNG seed', type=int, default=0)
        parser.add_argument('--critic-l2-reg', type=float, default=0)
        parser.add_argument(
            '--batch-size', type=int, default=64)  # per MPI worker
        parser.add_argument('--actor-lr', type=float, default=1e-4)
        parser.add_argument('--critic-lr', type=float, default=1e-3)
        boolean_flag(parser, 'popart', default=False)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--reward-scale', type=float, default=1.)
        parser.add_argument('--clip-norm', type=float, default=None)
        parser.add_argument(
            '--nb-epochs', type=int,
            default=500)  # with default settings, perform 1M steps total
        parser.add_argument('--nb-epoch-cycles', type=int, default=20)
        parser.add_argument(
            '--nb-train-steps', type=int,
            default=50)  # per epoch cycle and MPI worker
        parser.add_argument(
            '--nb-eval-steps', type=int,
            default=100)  # per epoch cycle and MPI worker
        parser.add_argument(
            '--nb-rollout-steps', type=int,
            default=100)  # per epoch cycle and MPI worker
        parser.add_argument(
            '--noise-type', type=str, default='adaptive-param_0.2'
        )  # choices are adaptive-param_xx, ou_xx, normal_xx, none
        parser.add_argument('--num-timesteps', type=int, default=None)
        boolean_flag(parser, 'evaluation', default=False)
        args = parser.parse_args()

        if args.num_timesteps is not None:
            assert (args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles
                    * args.nb_rollout_steps)
        dict_args = vars(args)
        del dict_args['num_timesteps']
        return dict_args

    args = parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    run(**args)


test_ddpg()
