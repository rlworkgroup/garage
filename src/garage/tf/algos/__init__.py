"""Tensorflow implementation of reinforcement learning algorithms."""
from garage.tf.algos.ddpg import DDPG
from garage.tf.algos.dqn import DQN
from garage.tf.algos.erwr import ERWR
from garage.tf.algos.npo import NPO
from garage.tf.algos.ppo import PPO
from garage.tf.algos.reps import REPS
from garage.tf.algos.rl2 import RL2
from garage.tf.algos.rl2ppo import RL2PPO
from garage.tf.algos.rl2trpo import RL2TRPO
from garage.tf.algos.td3 import TD3
from garage.tf.algos.te_npo import TENPO
from garage.tf.algos.te_ppo import TEPPO
from garage.tf.algos.tnpg import TNPG
from garage.tf.algos.trpo import TRPO
from garage.tf.algos.vpg import VPG

__all__ = [
    'DDPG',
    'DQN',
    'ERWR',
    'NPO',
    'PPO',
    'REPS',
    'RL2',
    'RL2PPO',
    'RL2TRPO',
    'TD3',
    'TNPG',
    'TRPO',
    'VPG',
    'TENPO',
    'TEPPO',
]
