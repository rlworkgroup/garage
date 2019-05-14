"""
This file overrides the train method of garage/algos/batch_polopt.py.
A socket has been added to notify the test of interrupted experiments about the
different stages in the experiment lifecycle.
"""

from multiprocessing.connection import Client

from dowel import logger, tabular

from garage.experiment import snapshotter
from garage.np.algos import BatchPolopt
from garage.plotter import Plotter
from tests.integration_tests.test_sigint import ExpLifecycle


class InstrumentedBatchPolopt(BatchPolopt):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo,
    etc.
    """

    def train(self):
        address = ('localhost', 6000)
        conn = Client(address)
        try:
            plotter = Plotter()
            if self.plot:
                plotter.init_plot(self.env, self.policy)
            conn.send(ExpLifecycle.START)
            self.start_worker()
            self.init_opt()
            for itr in range(self.current_itr, self.n_itr):
                with logger.prefix('itr #{} | '.format(itr)):
                    conn.send(ExpLifecycle.OBTAIN_SAMPLES)
                    paths = self.sampler.obtain_samples(itr)
                    conn.send(ExpLifecycle.PROCESS_SAMPLES)
                    samples_data = self.sampler.process_samples(itr, paths)
                    self.log_diagnostics(paths)
                    conn.send(ExpLifecycle.OPTIMIZE_POLICY)
                    self.optimize_policy(itr, samples_data)
                    logger.log('saving snapshot...')
                    params = self.get_itr_snapshot(itr, samples_data)
                    self.current_itr = itr + 1
                    params['algo'] = self
                    if self.store_paths:
                        params['paths'] = samples_data['paths']
                    snapshotter.save_itr_params(itr, params)
                    logger.log('saved')
                    logger.log(tabular)
                    if self.plot:
                        conn.send(ExpLifecycle.UPDATE_PLOT)
                        plotter.update_plot(self.policy, self.max_path_length)
                        if self.pause_for_plot:
                            input('Plotting evaluation run: Press Enter to '
                                  'continue...')

            conn.send(ExpLifecycle.SHUTDOWN)
            plotter.close()
            self.shutdown_worker()
        finally:
            conn.close()
