# Load the policy and the env in which it was trained
from garage.experiment import Snapshotter
import tensorflow as tf # optional, only for TensorFlow as we need a tf.Session
import fire

def run_rollouts(file, animated=True):
    snapshotter = Snapshotter()
    with tf.compat.v1.Session(): # optional, only for TensorFlow
        data = snapshotter.load(file)
    policy = data['algo'].policy
    env = data['env']

    # See what the trained policy can accomplish
    from garage.sampler.utils import rollout
    path = rollout(env, policy, animated=animated)
    print(path)

if __name__ == '__main__':
    fire.Fire(run_rollouts)