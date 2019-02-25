import time

import tensorflow as tf

from garage.misc.logger import logger
from garage.algos import BatchPolopt
from garage.tf.plotter import Plotter
from garage.tf.samplers import BatchSampler
from garage.tf.samplers import OffPolicyVectorizedSampler
from garage.tf.samplers import OnPolicyVectorizedSampler


class LocalRunner:
    def __init__(self, sess=None):
        self.sess = sess if sess else tf.Session()
        self.has_setup = False

    def __enter__(self):
        self.sess.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.__exit__(exc_type, exc_val, exc_tb)

    def setup(self, algo, env, sampler_cls=None, sampler_args=None):
        self.algo = algo
        self.env = env
        self.policy = self.algo.policy

        if sampler_args is None:
            sampler_args = {}

        if sampler_cls is None:
            if isinstance(algo, BatchPolopt):
                if self.policy.vectorized:
                    sampler_cls = OnPolicyVectorizedSampler
                else:
                    sampler_cls = BatchSampler
            else:
                sampler_cls = OffPolicyVectorizedSampler

        self.sampler = sampler_cls(algo, **sampler_args)

        self.initialize_tf_vars()
        self.has_setup = True

    def initialize_tf_vars(self):
        self.sess.run(
            tf.variables_initializer([
                v for v in tf.global_variables()
                if v.name.split(':')[0] in str(
                    self.sess.run(tf.report_uninitialized_variables()))
            ]))

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            self.plotter = Plotter(self.env, self.policy)
            self.plotter.start()

    def shutdown_worker(self):
        self.sampler.shutdown_worker()
        if self.plot:
            self.plotter.close()

    def obtain_samples(self, itr, batch_size):
        if self.n_epoch_cycles == 1:
            logger.log("Obtaining samples...")
        return self.sampler.obtain_samples(itr, batch_size)

    def save_snapshot(self, itr, paths=None):
        logger.log("Saving snapshot...")
        params = self.algo.get_itr_snapshot(itr, paths)
        if paths:
            params["paths"] = paths
        logger.save_itr_params(itr, params)
        logger.log("Saved")

    def log_diagnostics(self, pause_for_plot=False):
        logger.record_tabular('Time', time.time() - self.start_time)
        logger.record_tabular('ItrTime', time.time() - self.itr_start_time)
        logger.dump_tabular(with_prefix=False)
        if self.plot:
            self.plotter.update_plot(self.policy, self.algo.max_path_length)
            if pause_for_plot:
                input("Plotting evaluation run: Press Enter to " "continue...")

    def train(self,
              n_epochs,
              n_epoch_cycles=1,
              batch_size=None,
              plot=False,
              store_paths=False,
              pause_for_plot=False):

        assert self.has_setup, "Use Runner.setup() to setup runner " \
                               "before training."
        if batch_size is None:
            if isinstance(self.sampler, OffPolicyVectorizedSampler):
                batch_size = self.algo.max_path_length
            else:
                batch_size = 40 * self.algo.max_path_length

        self.n_epoch_cycles = n_epoch_cycles

        self.plot = plot
        self.start_worker()
        self.start_time = time.time()

        itr = 0
        for epoch in range(n_epochs):
            self.itr_start_time = time.time()
            paths = None
            with logger.prefix('epoch #%d | ' % epoch):
                for cycle in range(n_epoch_cycles):
                    paths = self.obtain_samples(itr, batch_size)
                    paths = self.sampler.process_samples(itr, paths)
                    self.algo.train_once(itr, paths)
                    itr += 1
            self.save_snapshot(epoch, paths if store_paths else None)
            self.log_diagnostics(pause_for_plot)

        self.shutdown_worker()
