"""Tensorflow implementation of reinforcement learning algorithms."""
from garage.tf.algos.batch_polopt import BatchPolopt
from garage.tf.algos.batch_polopt2 import BatchPolopt2
from garage.tf.algos.ddpg import DDPG
from garage.tf.algos.dqn import DQN
from garage.tf.algos.erwr import ERWR
from garage.tf.algos.npo import NPO
from garage.tf.algos.ppo import PPO
from garage.tf.algos.reps import REPS
from garage.tf.algos.rl2 import RL2
from garage.tf.algos.rl2npo import RL2NPO
from garage.tf.algos.rl2ppo import RL2PPO
from garage.tf.algos.td3 import TD3
from garage.tf.algos.tnpg import TNPG
from garage.tf.algos.trpo import TRPO
from garage.tf.algos.vpg import VPG

__all__ = [
    'BatchPolopt',
    'BatchPolopt2',
    'DDPG',
    'DQN',
    'ERWR',
    'NPO',
    'PPO',
    'REPS',
    'RL2',
    'RL2NPO',
    'RL2PPO',
    'TD3',
    'TNPG',
    'TRPO',
    'VPG',
]
