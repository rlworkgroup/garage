from garage.theano.algos.ddpg import DDPG
from garage.theano.algos.vpg import VPG
from garage.theano.algos.erwr import ERWR  # noqa: I100
from garage.theano.algos.npo import NPO
from garage.theano.algos.ppo import PPO
from garage.theano.algos.reps import REPS
from garage.theano.algos.tnpg import TNPG
from garage.theano.algos.trpo import TRPO

__all__ = ["DDPG", "VPG", "ERWR", "NPO", "PPO", "REPS", "TNPG", "TRPO"]
