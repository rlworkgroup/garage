#!/usr/bin/env python3
"""This is an example to add a simple VPG algorithm."""
import numpy as np
import tensorflow as tf

from garage import log_performance, TrajectoryBatch, wrap_experiment
from garage.envs import GarageEnv, PointEnv
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.misc import tensor_utils
from garage.sampler import RaySampler
from garage.tf.policies import GaussianMLPPolicy


# pylint: disable=missing-return-doc, missing-return-type-doc
# pylint: disable=missing-class-docstring, missing-function-docstring
class SimpleVPG:  # noqa: D101

    sampler_cls = RaySampler

    def __init__(self, env_spec, policy):
        self.env_spec = env_spec
        self.policy = policy
        self.max_path_length = 200
        self._discount = 0.99
        self.init_opt()

    def init_opt(self):  # noqa: D102
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

    def train(self, runner):  # noqa: D102
        for epoch in runner.step_epochs():
            samples = runner.obtain_samples(epoch)
            log_performance(
                epoch,
                TrajectoryBatch.from_trajectory_list(self.env_spec, samples),
                self._discount)
            self._train_once(samples)

    def _train_once(self, paths):
        obs = np.concatenate([path['observations'] for path in paths])
        actions = np.concatenate([path['actions'] for path in paths])
        returns = []
        for path in paths:
            returns.append(
                tensor_utils.discount_cumsum(path['rewards'], self._discount))
        returns = np.concatenate(returns)
        sess = tf.compat.v1.get_default_session()
        sess.run(self._train_op,
                 feed_dict={
                     self._observation: obs,
                     self._action: actions,
                     self._returns: returns,
                 })
        return np.mean(returns)

    def __getstate__(self):  # noqa: D105
        data = self.__dict__.copy()
        del data['_observation']
        del data['_action']
        del data['_returns']
        del data['_train_op']
        return data

    def __setstate__(self, state):  # noqa: D105
        self.__dict__ = state
        self.init_opt()


@wrap_experiment()
def debug_my_algorithm(ctxt=None):  # noqa: D103
    set_seed(100)
    with LocalTFRunner(ctxt) as runner:
        env = GarageEnv(PointEnv())
        policy = GaussianMLPPolicy(env.spec)
        algo = SimpleVPG(env.spec, policy)
        runner.setup(algo, env)
        runner.train(n_epochs=500, batch_size=4000, plot=True)


debug_my_algorithm()
