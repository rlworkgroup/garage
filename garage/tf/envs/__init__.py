from garage.tf.envs.base import TfEnv
from garage.tf.envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from garage.tf.envs.vec_env_executor import VecEnvExecutor


from garage.tf.spaces import Box
from garage.tf.spaces import Discrete
from garage.tf.spaces import Product


def to_tf_space(space):
    if isinstance(space, GymBox):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, GymDiscrete):
        return Discrete(space.n)
    elif isinstance(space, GymTuple):
        return Product(list(map(to_tf_space, space.spaces)))
    else:
        raise NotImplementedError