from collections import deque

import numpy as np
import tensorflow as tf

from garage.envs.util import configure_dims
from garage.envs.util import dims_to_shapes
import garage.misc.logger as logger
from garage.misc.overrides import overrides
from garage.replay_buffer import HerReplayBuffer, RegularReplayBuffer
from garage.replay_buffer.base import Buffer
from garage.replay_buffer.her_replay_buffer import make_her_sample
from garage.tf.algos.batch_polopt import BatchPolopt


class OffPolicyBatchPolopt(BatchPolopt):
    def __init__(self,
                 env,
                 policy,
                 qf,
                 replay_buffer_type=Buffer.REGULAR,
                 use_target=False,
                 n_epochs=500,
                 n_epoch_cycles=20,
                 max_path_length=100,
                 n_train_steps=50,
                 buffer_batch_size=64,
                 replay_k=4,
                 replay_buffer_size=int(1e6),
                 min_buffer_size=int(1e4),
                 reward_scale=1.,
                 rollout_batch_size=1,
                 **kwargs):
        self.input_dims = configure_dims(env)
        self.n_epochs = n_epochs
        self.n_epoch_cycles = n_epoch_cycles
        self.n_train_steps = n_train_steps
        self.buffer_batch_size = buffer_batch_size
        self.use_target = use_target
        self.replay_buffer_type = replay_buffer_type
        self.replay_k = replay_k
        self.replay_buffer_size = replay_buffer_size
        self.min_buffer_size = min_buffer_size
        self.reward_scale = reward_scale
        self.rollout_batch_size = rollout_batch_size
        self.success_history = deque(maxlen=100)
        self.evaluate = False
        input_shapes = dims_to_shapes(self.input_dims)
        if self.replay_buffer_type == Buffer.HER:
            buffer_shapes = {
                key: (max_path_length + 1
                      if key == "observation" or key == "achieved_goal" else
                      max_path_length, *input_shapes[key])
                for key, val in input_shapes.items()
            }

            self.replay_buffer = HerReplayBuffer(
                buffer_shapes=buffer_shapes,
                size_in_transitions=self.replay_buffer_size,
                time_horizon=max_path_length,
                sample_transitions=make_her_sample(self.replay_k,
                                                   env.compute_reward))
        elif self.replay_buffer_type == Buffer.REGULAR:
            buffer_shapes = {
                key: (max_path_length, *input_shapes[key])
                for key, val in input_shapes.items()
            }

            self.replay_buffer = RegularReplayBuffer(
                buffer_shapes=buffer_shapes,
                size_in_transitions=self.replay_buffer_size,
                time_horizon=max_path_length)
        else:
            raise NotImplementedError("Unknown replay_buffer_type")
        input_include_goal = True if (
            self.replay_buffer_type == Buffer.HER) else False
        super(OffPolicyBatchPolopt, self).__init__(
            env=env,
            policy=policy,
            baseline=qf,
            max_path_length=max_path_length,
            input_include_goal=input_include_goal,
            **kwargs)

    @overrides
    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        sess.run(tf.global_variables_initializer())
        self.start_worker(sess)

        if self.use_target:
            self.f_init_target()

        episode_rewards = []
        episode_policy_losses = []
        episode_qf_losses = []
        epoch_ys = []
        epoch_qs = []

        for epoch in range(self.n_epochs):
            self.success_history.clear()
            with logger.prefix('epoch #%d | ' % epoch):
                for epoch_cycle in range(self.n_epoch_cycles):
                    paths = self.obtain_samples(epoch)
                    samples_data = self.process_samples(epoch, paths)
                    episode_rewards.extend(
                        samples_data["undiscounted_returns"])
                    self.success_history.extend(
                        samples_data["success_history"])
                    self.log_diagnostics(paths)
                    for train_itr in range(self.n_train_steps):
                        if (self.replay_buffer.n_transitions_stored >=
                                self.min_buffer_size) or (
                                    self.replay_buffer_type == Buffer.HER):
                            self.evaluate = True
                            qf_loss, y, q, policy_loss = self.optimize_policy(
                                epoch, samples_data)

                            episode_policy_losses.append(policy_loss)
                            episode_qf_losses.append(qf_loss)
                            epoch_ys.append(y)
                            epoch_qs.append(q)

                    if self.plot:
                        self.plotter.update_plot(self.policy,
                                                 self.max_path_length)
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                  "continue...")

                logger.log("Training finished")
                logger.log("Saving snapshot #{}".format(epoch))
                params = self.get_itr_snapshot(epoch, samples_data)
                logger.save_itr_params(epoch, params)
                logger.log("Saved")
                if self.evaluate:
                    logger.record_tabular('Epoch', epoch)
                    logger.record_tabular('AverageReturn',
                                          np.mean(episode_rewards))
                    logger.record_tabular('StdReturn', np.std(episode_rewards))
                    logger.record_tabular('Policy/AveragePolicyLoss',
                                          np.mean(episode_policy_losses))
                    logger.record_tabular('QFunction/AverageQFunctionLoss',
                                          np.mean(episode_qf_losses))
                    logger.record_tabular('QFunction/AverageQ',
                                          np.mean(epoch_qs))
                    logger.record_tabular('QFunction/MaxQ', np.max(epoch_qs))
                    logger.record_tabular('QFunction/AverageAbsQ',
                                          np.mean(np.abs(epoch_qs)))
                    logger.record_tabular('QFunction/AverageY',
                                          np.mean(epoch_ys))
                    logger.record_tabular('QFunction/MaxY', np.max(epoch_ys))
                    logger.record_tabular('QFunction/AverageAbsY',
                                          np.mean(np.abs(epoch_ys)))
                    if self.replay_buffer_type == Buffer.HER:
                        logger.record_tabular('AverageSuccessRate',
                                              np.mean(self.success_history))

                # Uncomment the following if you want to calculate the average
                # in each epoch, better uncomment when self.use_her is True
                # episode_rewards = []
                # episode_policy_losses = []
                # episode_qf_losses = []
                # epoch_ys = []
                # epoch_qs = []

                logger.dump_tabular(with_prefix=False)

        self.shutdown_worker()
        if created_session:
            sess.close()

    @property
    def off_policy(self):
        return True
