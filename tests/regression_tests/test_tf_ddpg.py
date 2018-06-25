"""
This script creates a regression test over garage-DDPG and baselines-DDPG.

It get Mujoco1M benchmarks from baselines benchmark, and test each task in
its trail times on garage-DDPG model and baselines-DDPG model.
"""
import datetime
import os.path as osp

from baselines import logger
from baselines.bench import benchmarks
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
import baselines.ddpg.training as training
from baselines.logger import configure
import gym
from mpi4py import MPI
import numpy as np
import tensorflow as tf

from garage.misc import logger as garage_logger
from garage.tf.algos import DDPG
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction


def test_ddpg():
    """
    A test function to compare benchmarks of DDPG over seven mujoco models
    between garage and baselines version.
    :return:
    """
    # Load Mujoco1M tasks
    mujoco1m = benchmarks.get_benchmark("Mujoco1M")

    # Set up params for garage ddpg
    params = {
        "InvertedDoublePendulum-v2": {
            "actor_hidden_sizes": [64],
            "critic_hidden_sizes": [64],
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "replay_buffer_size": 1e4,
            "min_buffer_size": 1e3,
            "target_update_tau": 1e-3,
        },
        "InvertedPendulum-v2": {
            "actor_hidden_sizes": [64, 64],
            "critic_hidden_sizes": [64, 64],
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "replay_buffer_size": 1e4,
            "min_buffer_size": 1e3,
            "target_update_tau": 1e-3,
        },
        "HalfCheetah-v2": {
            "actor_hidden_sizes": [64, 64],
            "critic_hidden_sizes": [64, 64],
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "replay_buffer_size": 1e6,
            "min_buffer_size": 1e4,
            "target_update_tau": 1e-3,
        },
        "Hopper-v2": {
            "actor_hidden_sizes": [128, 128],
            "critic_hidden_sizes": [128, 128],
            "actor_lr": 1e-5,
            "critic_lr": 1e-3,
            "replay_buffer_size": 1e6,
            "min_buffer_size": 1e4,
            "target_update_tau": 1e-3,
        },
        "Walker2d-v2": {
            "actor_hidden_sizes": [64, 64],
            "critic_hidden_sizes": [64, 64],
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "replay_buffer_size": 1e6,
            "min_buffer_size": 1e4,
            "target_update_tau": 1e-2,
        },
        "Reacher-v2": {
            "actor_hidden_sizes": [64, 64],
            "critic_hidden_sizes": [64, 64],
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "replay_buffer_size": 1e6,
            "min_buffer_size": 1e4,
            "target_update_tau": 1e-2,
        },
        "Swimmer-v2": {
            "actor_hidden_sizes": [64, 64],
            "critic_hidden_sizes": [64, 64],
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "replay_buffer_size": 1e6,
            "min_buffer_size": 1e4,
            "target_update_tau": 1e-2,
        },
    }

    for task in mujoco1m["tasks"]:
        env_id = task["env_id"]
        env = gym.make(env_id)
        garage_ddpg_params = params[env_id]
        for i in range(task["trials"]):
            env.reset()
            seed = np.random.randint(10)

            # Run garage ddpg
            env.seed(seed)

            actor_hidden_sizes = garage_ddpg_params["actor_hidden_sizes"]
            critic_hidden_sizes = garage_ddpg_params["critic_hidden_sizes"]
            actor_lr = garage_ddpg_params["actor_lr"]
            critic_lr = garage_ddpg_params["critic_lr"]
            replay_buffer_size = garage_ddpg_params["replay_buffer_size"]
            min_buffer_size = garage_ddpg_params["min_buffer_size"]
            target_update_tau = garage_ddpg_params["target_update_tau"]

            action_noise = OUStrategy(env, sigma=0.2)

            actor_net = ContinuousMLPPolicy(
                env_spec=env,
                name="Actor",
                hidden_sizes=actor_hidden_sizes,
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=tf.nn.tanh)

            critic_net = ContinuousMLPQFunction(
                env_spec=env,
                name="Critic",
                hidden_sizes=critic_hidden_sizes,
                hidden_nonlinearity=tf.nn.relu)

            ddpg = DDPG(
                env,
                actor=actor_net,
                critic=critic_net,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                plot=False,
                target_update_tau=target_update_tau,
                n_epochs=500,
                n_epoch_cycles=20,
                n_rollout_steps=100,
                n_train_steps=50,
                discount=0.9,
                replay_buffer_size=int(replay_buffer_size),
                min_buffer_size=int(min_buffer_size),
                exploration_strategy=action_noise,
                actor_optimizer=tf.train.AdamOptimizer,
                critic_optimizer=tf.train.AdamOptimizer)

            rand_dir = datetime.datetime.now().strftime(
                "garage-%Y-%m-%d-%H-%M-%S-%f")
            log_dir = "./ddpg_benchmark/" + rand_dir + "/" + task["env_id"] \
                      + "/trail_" + str(i) + "_seed_" + str(seed) + "/"
            tabular_log_file = osp.join(log_dir, "progress.csv")
            tensorboard_log_dir = osp.join(log_dir, "progress")
            garage_logger.add_tabular_output(tabular_log_file)
            garage_logger.set_tensorboard_dir(tensorboard_log_dir)

            ddpg.train()

            # Run baselines ddpg
            rank = MPI.COMM_WORLD.Get_rank()
            env.seed(seed)
            nb_actions = env.action_space.shape[-1]
            layer_norm = False
            action_noise = OrnsteinUhlenbeckActionNoise(
                mu=np.zeros(nb_actions),
                sigma=float(0.2) * np.ones(nb_actions))
            memory = Memory(
                limit=int(1e6),
                action_shape=env.action_space.shape,
                observation_shape=env.observation_space.shape)
            critic = Critic(layer_norm=layer_norm)
            actor = Actor(nb_actions, layer_norm=layer_norm)

            rand_dir = datetime.datetime.now().strftime(
                "baselines-%Y-%m-%d-%H-%M-%S-%f")
            log_dir = "./ddpg_benchmark/" + rand_dir + "/" + task["env_id"] \
                      + "/trail_" + str(i) + "_seed_" + str(seed) + "/"
            configure(dir=log_dir)
            logger.info('rank {}: seed={}, logdir={}'.format(
                rank, seed, logger.get_dir()))

            training.train(
                env=env,
                eval_env=None,
                param_noise=None,
                action_noise=action_noise,
                actor=actor,
                critic=critic,
                memory=memory,
                nb_epochs=500,
                nb_epoch_cycles=20,
                render_eval=False,
                reward_scale=1.,
                render=False,
                normalize_returns=False,
                normalize_observations=False,
                critic_l2_reg=0,
                actor_lr=1e-4,
                critic_lr=1e-3,
                popart=False,
                gamma=0.99,
                clip_norm=None,
                nb_train_steps=50,
                nb_rollout_steps=100,
                nb_eval_steps=100,
                batch_size=64)


test_ddpg()
