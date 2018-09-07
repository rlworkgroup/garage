import time

import tensorflow as tf

import garage.misc.logger as logger
from garage.misc.overrides import overrides
from garage.tf.algos.batch_polopt import BatchPolopt


class OnPolicyBatchPolopt(BatchPolopt):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo,
    etc.
    """

    def __init__(self,
                 n_itr=500,
                 start_itr=0,
                 gae_lambda=1,
                 batch_size=5000,
                 center_adv=True,
                 positive_adv=False,
                 whole_paths=True,
                 fixed_horizon=False,
                 **kwargs):
        """
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param center_adv: Whether to rescale the advantages so that they have
         mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are
         always positive. When used in conjunction with center_adv the
         advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        super(OnPolicyBatchPolopt, self).__init__(**kwargs)

    @overrides
    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        sess.run(tf.global_variables_initializer())
        self.start_worker(sess)
        start_time = time.time()
        last_average_return = None
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                paths = self.obtain_samples(itr)
                logger.log("Processing samples...")
                samples_data = self.process_samples(itr, paths)
                last_average_return = samples_data["average_return"]
                logger.log("Logging diagnostics...")
                self.log_diagnostics(paths)
                logger.log("Optimizing policy...")
                self.optimize_policy(itr, samples_data)
                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.plotter.update_plot(self.policy, self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")

        self.shutdown_worker()
        if created_session:
            sess.close()
        return last_average_return

    @property
    def off_policy(self):
        return False
