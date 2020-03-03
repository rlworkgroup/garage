"""Offline meta-testing helper."""
import argparse
import math
import os
import re
import sys

from dowel import CsvOutput, logger, StdOutput, tabular
import pandas as pd

from garage.experiment import LocalRunner, SnapshotConfig
from garage.experiment import MetaEvaluator, Snapshotter
from garage.experiment.task_sampler import FullSetTaskSampler


class OfflineTorchMetaTestHelper:
    """Offline meta-testing helper for PyTorch Meta-ML algorithms.

    This helper does offline meta-testing evaluation for an arbitrary PyTorch
    Meta-ML algorithms. It works by loading policy from snapshot then doing
    adaptation and evaluation.

    Meta-testing result will be saved in meta-test.csv file in the folder
    where snapshot is loaded.

    This helper provides a command line interface through read_cmd().

    Args:
        meta_task_cls (garage.envs.GarageEnv): A meta environment constructor.
            Can be either a class or callable.
            This environment should support sample_tasks(num_task) and
            set_task(task).
        max_path_length (int): Maximum path length.
        adapt_rollout_per_task (int): Number of rollouts to sample for
            adaptation.
        test_rollout_per_task (int): Number of rollouts to sample from an
            adapted policy for evaluation.

    """

    def __init__(self,
                 meta_task_cls,
                 max_path_length=150,
                 adapt_rollout_per_task=10,
                 test_rollout_per_task=10):

        self.meta_task_cls = meta_task_cls
        self.max_path_length = max_path_length
        self.adapt_rollout_per_task = adapt_rollout_per_task
        self.test_rollout_per_task = test_rollout_per_task

    @classmethod
    def read_cmd(cls, env_cls):
        """Read command line arguments and do meta-testing.

        Args:
            env_cls (garage.envs.GarageEnv): A meta environment constructor.
                Can be either a class or callable.
                This environment should support sample_tasks(num_task) and
                set_task(task).

        """
        logger.add_output(StdOutput())

        parser = argparse.ArgumentParser()
        parser.add_argument('folder', nargs='+')
        # Adaptation parameters
        parser.add_argument('--adapt-rollouts',
                            nargs='?',
                            default=10,
                            type=int)
        parser.add_argument('--test-rollouts', nargs='?', default=10, type=int)
        parser.add_argument('--max-path-length',
                            nargs='?',
                            default=100,
                            type=int)
        # Number of workers
        parser.add_argument('--parallel', nargs='?', default=0, type=int)
        # Skip iteration that has existing meta-testing result.
        parser.add_argument('--skip-exist', action='store_true', default=True)
        # Merge all meta-testing result to meta-test.csv
        parser.add_argument('--merge', action='store_true', default=True)
        # Skip some iterations.
        # e.g. stride=3 sample 1 iteration every 3 iterations.
        parser.add_argument('--stride', default=1, type=int)

        args = parser.parse_args()
        meta_train_dirs = args.folder
        workers = args.parallel
        adapt_rollout_per_task = args.adapt_rollouts
        test_rollout_per_task = args.test_rollouts
        max_path_length = args.max_path_length
        skip_existing = args.skip_exist
        to_merge = args.merge
        stride = args.stride

        helper = cls(meta_task_cls=env_cls,
                     max_path_length=max_path_length,
                     adapt_rollout_per_task=adapt_rollout_per_task,
                     test_rollout_per_task=test_rollout_per_task)

        helper.test_many_folders(folders=meta_train_dirs,
                                 workers=workers,
                                 skip_existing=skip_existing,
                                 to_merge=to_merge,
                                 stride=stride)

    @classmethod
    def _get_tested_itrs(cls, meta_train_dir):
        """Get iterations that have been tested.

        Args:
            meta_train_dir (str): Folder where snapshots are saved.

        Returns:
            list[int]: List of iterations that have been tested.

        """
        files = [f for f in os.listdir(meta_train_dir) if f.endswith('.csv')]
        if not files:
            return []

        itrs = []
        for file in files:
            nums = re.findall(r'\d+', file)
            if nums:
                itrs.append(int(nums[0]))
        itrs.sort()

        return itrs

    @classmethod
    def _merge_csv(cls, folder, itrs):
        """Merge per-iteration meta-test csv into a single meta-test.csv.

        Args:
            folder (str): Folder where csv files resides.
            itrs (list[int]): List of iterations to merge.

        """
        merged_file = os.path.join(folder, 'meta-test.csv')
        files_to_merge = [
            os.path.join(folder, 'meta-test-itr_{}.csv'.format(itr))
            for itr in itrs
        ]

        if os.path.isfile(merged_file):
            files_to_merge.append(merged_file)

        merged_csv = pd.concat([pd.read_csv(f) for f in files_to_merge],
                               sort=True)
        merged_csv.sort_values(by=['Iteration'])
        merged_csv.to_csv(merged_file, index=False)

        logger.log('Merged iteration {} into {}'.format(
            ', '.join([str(itr) for itr in itrs]), merged_file))

    def test_one_folder(self, meta_train_dir, itrs):
        """Do meta-testing for one folder.

        Args:
            meta_train_dir (str): Experiment folder where snapshots are saved.
            itrs (list[int]): List of iterations to test.

        """
        snapshot_config = SnapshotConfig(snapshot_dir=meta_train_dir,
                                         snapshot_mode='all',
                                         snapshot_gap=1)

        runner = LocalRunner(snapshot_config=snapshot_config)
        meta_sampler = FullSetTaskSampler(self.meta_task_cls)
        runner.restore(meta_train_dir)

        meta_evaluator = MetaEvaluator(
            runner,
            test_task_sampler=meta_sampler,
            max_path_length=self.max_path_length,
            n_test_tasks=meta_sampler.n_tasks,
            n_exploration_traj=self.adapt_rollout_per_task,
            n_evaluation_traj=self.test_rollout_per_task,
            prefix='')

        for itr in itrs:
            log_filename = os.path.join(meta_train_dir,
                                        'meta-test-itr_{}.csv'.format(itr))
            logger.add_output(CsvOutput(log_filename))
            logger.log('Writing into {}'.format(log_filename))

            runner.restore(meta_train_dir, from_epoch=itr)
            # pylint: disable=protected-access
            meta_evaluator.evaluate(runner._algo)
            tabular.record('Iteration', runner._stats.total_epoch)
            tabular.record('TotalEnvSteps', runner._stats.total_env_steps)
            logger.log(tabular)
            logger.dump_output_type(CsvOutput)
            logger.remove_output_type(CsvOutput)

    def test_many_folders(self, folders, workers, skip_existing, to_merge,
                          stride):
        """Do meta-testing for multiple folders.

        Args:
            folders (list[str]): Experiment folders where snapshots are saved.
            workers (int): Number of parallel workers.
            skip_existing (bool): If True, tested iterations will be skipped.
            to_merge (bool): If True, merge per-iteration meta-test csv to a
                single meta-test.csv.
            stride (int): Stride between two tested iterations. If set to 1,
                all available iterations will be tested.

        """
        for meta_train_dir in folders:
            itrs = Snapshotter.get_available_itrs(meta_train_dir)
            tested_itrs = self._get_tested_itrs(meta_train_dir)

            if skip_existing:
                itrs = [itr for itr in itrs if itr not in tested_itrs]

            if stride > 1:
                itrs = itrs[::stride]

            if workers == 0:
                self.test_one_folder(meta_train_dir, itrs)
            else:
                bite_size = math.ceil(len(itrs) / workers)
                bites = [
                    itrs[i * bite_size:(i + 1) * bite_size]
                    for i in range(workers)
                ]

                children = []
                for bite_itrs in bites:
                    if len(bite_itrs) == 0:
                        continue
                    pid = os.fork()
                    if pid == 0:
                        # In child process
                        self.test_one_folder(meta_train_dir, bite_itrs)
                        sys.exit()
                    else:
                        # In parent process
                        children.append(pid)

                for child in children:
                    os.waitpid(child, 0)

            if to_merge:
                self._merge_csv(meta_train_dir, itrs)
