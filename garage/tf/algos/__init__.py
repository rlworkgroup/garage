from garage.tf.algos.batch_polopt import BatchPolopt
from garage.tf.algos.ddpg import DDPG
from garage.tf.algos.dqn import DQN
from garage.tf.algos.erwr import ERWR
from garage.tf.algos.npo import NPO
from garage.tf.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.tf.algos.ppo import PPO
from garage.tf.algos.reps import REPS
from garage.tf.algos.tnpg import TNPG
from garage.tf.algos.trpo import TRPO
from garage.tf.algos.vpg import VPG

__all__ = [
    "OffPolicyRLAlgorithm",
    "BatchPolopt",
    "DDPG",
    "DQN",
    "ERWR",
    "NPO",
    "PPO",
    "REPS",
    "TNPG",
    "TRPO",
    "VPG",
]
