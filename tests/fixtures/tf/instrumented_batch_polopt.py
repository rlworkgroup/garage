"""This file overrides the train method of garage/tf/algos/batch_polopt.py.

A socket has been added to notify the test of interrupted experiments about the
different stages in the experiment lifecycle.
"""

from multiprocessing.connection import Client
import time

from dowel import logger, tabular
import tensorflow as tf

from garage.experiment import snapshotter
from garage.tf.algos import BatchPolopt
from tests.integration_tests.test_sigint import ExpLifecycle


class InstrumentedBatchPolopt(BatchPolopt):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo,
    etc.
    """

    def train(self, sess=None):
        address = ('localhost', 6000)
        conn = Client(address)
        last_average_return = None
        try:
            created_session = True if (sess is None) else False
            if sess is None:
                sess = tf.compat.v1.Session()
                sess.__enter__()

            sess.run(tf.compat.v1.global_variables_initializer())
            conn.send(ExpLifecycle.START)
            self.start_worker(sess)
            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    logger.log('Obtaining samples...')
                    conn.send(ExpLifecycle.OBTAIN_SAMPLES)
                    paths = self.obtain_samples(itr)
                    logger.log('Processing samples...')
                    conn.send(ExpLifecycle.PROCESS_SAMPLES)
                    samples_data = self.process_samples(itr, paths)
                    last_average_return = samples_data['average_return']
                    logger.log('Logging diagnostics...')
                    self.log_diagnostics(paths)
                    logger.log('Optimizing policy...')
                    conn.send(ExpLifecycle.OPTIMIZE_POLICY)
                    self.optimize_policy(itr, samples_data)
                    logger.log('Saving snapshot...')
                    params = self.get_itr_snapshot(itr)
                    if self.store_paths:
                        params['paths'] = samples_data['paths']
                    snapshotter.save_itr_params(itr, params)
                    logger.log('Saved')
                    tabular.record('Time', time.time() - start_time)
                    tabular.record('ItrTime', time.time() - itr_start_time)
                    logger.log(tabular)
                    if self.plot:
                        conn.send(ExpLifecycle.UPDATE_PLOT)
                        self.plotter.update_plot(self.policy,
                                                 self.max_path_length)
                        if self.pause_for_plot:
                            input('Plotting evaluation run: Press Enter to '
                                  'continue...')

            conn.send(ExpLifecycle.SHUTDOWN)
            self.shutdown_worker()
            if created_session:
                sess.close()
        finally:
            conn.close()
        return last_average_return
