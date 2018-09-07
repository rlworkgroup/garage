from garage.tf.algos.batch_polopt import BatchPolopt
from garage.tf.algos.ddpg import DDPG
from garage.tf.algos.npo import NPO
from garage.tf.algos.off_policy_batch_polopt import OffPolicyBatchPolopt
from garage.tf.algos.on_policy_batch_polopt import OnPolicyBatchPolopt
from garage.tf.algos.ppo import PPO
from garage.tf.algos.trpo import TRPO
from garage.tf.algos.vpg import VPG

__all__ = [
    "OffPolicyBatchPolopt", "OnPolicyBatchPolopt", "DDPG", "NPO", "PPO",
    "TRPO", "VPG"
]
