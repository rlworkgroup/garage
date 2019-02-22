import time

from garage.misc.logger import logger
from garage.tf.plotter import Plotter
from garage.tf.samplers import BatchSampler
from garage.tf.samplers import OnPolicyVectorizedSampler


class LocalRunner:
    def __init__(self, algo, env, sampler=None):
        self.algo = algo
        self.env = self.env
        self.policy = self.algo.policy

        if sampler is None:
            if self.policy.vectorized:
                sampler = OnPolicyVectorizedSampler(algo)
            else:
                sampler = BatchSampler(algo)

        self.sampler = sampler

        self.start_time = None
        self.plot = False

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            self.plotter = Plotter(self.env, self.policy)
            self.plotter.start()

    def shutdown_worker(self):
        self.sampler.shutdown_worker()
        if self.plot:
            self.plotter.close()

    def obtain_samples(self, itr):
        logger.log("Obtaining samples...")
        return self.sampler.obtain_samples(itr)

    def save_snapshot(self, itr, paths=None):
        logger.log("Saving snapshot...")
        params = self.algo.get_itr_snapshot(itr)
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

    def train(self, n_itr, plot=False, store_paths=False,
              pause_for_plot=False):
        self.plot = plot

        self.algo.initialize()
        self.start_worker()

        self.start_time = time.time()

        for itr in range(0, n_itr):
            self.itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                paths = self.obtain_samples(itr)
                self.algo.train_once(paths)
                self.save_snapshot(itr, paths if store_paths else None)
                self.log_diagnostics(pause_for_plot)

        self.shutdown_worker()
