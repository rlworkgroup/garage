#!/usr/bin/env python3
"""This is an example to add a simple VPG algorithm."""
import numpy as np
import tensorflow as tf

from garage import EpisodeBatch, log_performance, wrap_experiment
from garage.envs import PointEnv
from garage.experiment.deterministic import set_seed
from garage.np import discount_cumsum
from garage.sampler import LocalSampler
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer


class SimpleVPG:
    """Simple Vanilla Policy Gradient.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.StochasticPolicy): Policy.
        sampler (garage.sampler.Sampler): Sampler.

    """

    def __init__(self, env_spec, policy, sampler):
        self.env_spec = env_spec
        self.policy = policy
        self._sampler = sampler
        self.max_episode_length = env_spec.max_episode_length
        self._discount = 0.99
        self.init_opt()

    def init_opt(self):
        """Initialize optimizer and build computation graph."""
        observation_dim = self.policy.observation_space.flat_dim
        action_dim = self.policy.action_space.flat_dim
        with tf.name_scope('inputs'):
            self._observation = tf.compat.v1.placeholder(
                tf.float32, shape=[None, observation_dim], name='observation')
            self._action = tf.compat.v1.placeholder(tf.float32,
                                                    shape=[None, action_dim],
                                                    name='action')
            self._returns = tf.compat.v1.placeholder(tf.float32,
                                                     shape=[None],
                                                     name='return')
        policy_dist = self.policy.build(self._observation, name='policy').dist
        with tf.name_scope('loss'):
            ll = policy_dist.log_prob(self._action, name='log_likelihood')
            loss = -tf.reduce_mean(ll * self._returns)
        with tf.name_scope('train'):
            self._train_op = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(
                loss)

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer.

        """
        for epoch in trainer.step_epochs():
            samples = trainer.obtain_samples(epoch)
            log_performance(epoch,
                            EpisodeBatch.from_list(self.env_spec, samples),
                            self._discount)
            self._train_once(samples)

    def _train_once(self, samples):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            samples (list[dict]): A list of collected samples.

        Returns:
            numpy.float64: Average return.

        """
        obs = np.concatenate([path['observations'] for path in samples])
        actions = np.concatenate([path['actions'] for path in samples])
        returns = []
        for path in samples:
            returns.append(discount_cumsum(path['rewards'], self._discount))
        returns = np.concatenate(returns)
        sess = tf.compat.v1.get_default_session()
        sess.run(self._train_op,
                 feed_dict={
                     self._observation: obs,
                     self._action: actions,
                     self._returns: returns,
                 })
        return np.mean(returns)

    def __getstate__(self):
        """Parameters to save in snapshot.

        Returns:
            dict: Parameters to save.

        """
        data = self.__dict__.copy()
        del data['_observation']
        del data['_action']
        del data['_returns']
        del data['_train_op']
        return data

    def __setstate__(self, state):
        """Parameters to restore from snapshot.

        Args:
            state (dict): Parameters to restore from.

        """
        self.__dict__ = state
        self.init_opt()


@wrap_experiment
def tutorial_vpg(ctxt=None):
    """Train VPG with PointEnv environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.

    """
    set_seed(100)
    with TFTrainer(ctxt) as trainer:
        env = PointEnv(max_episode_length=200)
        policy = GaussianMLPPolicy(env.spec)
        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True)
        algo = SimpleVPG(env.spec, policy, sampler)
        trainer.setup(algo, env)
        trainer.train(n_epochs=200, batch_size=4000)


tutorial_vpg()
