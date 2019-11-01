#!/usr/bin/env python3
"""Run an experiment triggered by `run_experiment()` in `experiment.py`."""
import argparse
import ast
import base64
import datetime
import enum
import json
import os
import pathlib
import pickle
import sys
import uuid

import cloudpickle
import dateutil.tz
import dowel
from dowel import logger
import psutil

from garage.experiment import deterministic, SnapshotConfig
import garage.plotter
from garage.sampler import parallel_sampler
import garage.tf.plotter


def is_iterable(obj):
    """Return if instance is iterable."""
    return isinstance(obj, str) or getattr(obj, '__iter__', False)


def run_experiment(argv):
    """Run experiment."""
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    # avoid name clashes when running distributed jobs
    rand_id = str(uuid.uuid4())[:5]
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    default_exp_name = 'experiment_%s_%s' % (timestamp, rand_id)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_parallel',
        type=int,
        default=1,
        help=('Number of parallel workers to perform rollouts. '
              "0 => don't start any workers"))
    parser.add_argument('--exp_name',
                        type=str,
                        default=default_exp_name,
                        help='Name of the experiment.')
    parser.add_argument('--log_dir',
                        type=str,
                        default=None,
                        help='Path to save the log and iteration snapshot.')
    parser.add_argument('--snapshot_mode',
                        type=str,
                        default='last',
                        help='Mode to save the snapshot. Can be either "all" '
                        '(all iterations will be saved), "last" (only '
                        'the last iteration will be saved), "gap" (every'
                        '`snapshot_gap` iterations are saved), or "none" '
                        '(do not save snapshots)')
    parser.add_argument('--snapshot_gap',
                        type=int,
                        default=1,
                        help='Gap between snapshot iterations.')
    parser.add_argument(
        '--resume_from_dir',
        type=str,
        default=None,
        help='Directory of the pickle file to resume experiment from.')
    parser.add_argument('--resume_from_epoch',
                        type=str,
                        default=None,
                        help='Index of iteration to restore from. '
                        'Can be "first", "last" or a number. '
                        'Not applicable when snapshot_mode="last"')
    parser.add_argument('--tabular_log_file',
                        type=str,
                        default='progress.csv',
                        help='Name of the tabular log file (in csv).')
    parser.add_argument('--text_log_file',
                        type=str,
                        default='debug.log',
                        help='Name of the text log file (in pure text).')
    parser.add_argument('--tensorboard_step_key',
                        type=str,
                        default=None,
                        help='Name of the step key in tensorboard_summary.')
    parser.add_argument('--params_log_file',
                        type=str,
                        default='params.json',
                        help='Name of the parameter log file (in json).')
    parser.add_argument('--variant_log_file',
                        type=str,
                        default='variant.json',
                        help='Name of the variant log file (in json).')
    parser.add_argument('--plot',
                        type=ast.literal_eval,
                        default=False,
                        help='Whether to plot the iteration results')
    parser.add_argument(
        '--log_tabular_only',
        type=ast.literal_eval,
        default=False,
        help='Print only the tabular log information (in a horizontal format)')
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='Random seed for numpy')
    parser.add_argument('--args_data',
                        type=str,
                        help='Pickled data for objects')
    parser.add_argument('--variant_data',
                        type=str,
                        help='Pickled data for variant configuration')

    args = parser.parse_args(argv[1:])

    if args.seed is not None:
        deterministic.set_seed(args.seed)

    if args.n_parallel > 0:
        parallel_sampler.initialize(n_parallel=args.n_parallel)
        if args.seed is not None:
            parallel_sampler.set_seed(args.seed)

    if not args.plot:
        garage.plotter.Plotter.disable()
        garage.tf.plotter.Plotter.disable()

    if args.log_dir is None:
        log_dir = os.path.join(os.path.join(os.getcwd(), 'data'),
                               args.exp_name)
    else:
        log_dir = args.log_dir

    tabular_log_file = os.path.join(log_dir, args.tabular_log_file)
    text_log_file = os.path.join(log_dir, args.text_log_file)
    params_log_file = os.path.join(log_dir, args.params_log_file)

    if args.variant_data is not None:
        variant_data = pickle.loads(base64.b64decode(args.variant_data))
        variant_log_file = os.path.join(log_dir, args.variant_log_file)
        dump_variant(variant_log_file, variant_data)
    else:
        variant_data = None

    log_parameters(params_log_file, args)

    logger.add_output(dowel.TextOutput(text_log_file))
    logger.add_output(dowel.CsvOutput(tabular_log_file))
    logger.add_output(dowel.TensorBoardOutput(log_dir))
    logger.add_output(dowel.StdOutput())

    logger.push_prefix('[%s] ' % args.exp_name)

    snapshot_config = SnapshotConfig(snapshot_dir=log_dir,
                                     snapshot_mode=args.snapshot_mode,
                                     snapshot_gap=args.snapshot_gap)

    method_call = cloudpickle.loads(base64.b64decode(args.args_data))
    try:
        method_call(snapshot_config, variant_data, args.resume_from_dir,
                    args.resume_from_epoch)
    except BaseException:
        children = garage.plotter.Plotter.get_plotters()
        children += garage.tf.plotter.Plotter.get_plotters()
        if args.n_parallel > 0:
            children += [parallel_sampler]
        child_proc_shutdown(children)
        raise

    logger.remove_all()
    logger.pop_prefix()


def child_proc_shutdown(children):
    """Shut down children processes."""
    run_exp_proc = psutil.Process()
    alive = run_exp_proc.children(recursive=True)
    for proc in alive:
        if any([
                'multiprocessing.semaphore_tracker' in cmd
                for cmd in proc.cmdline()
        ]):
            alive.remove(proc)

    for c in children:
        c.close()
    max_retries = 5
    for _ in range(max_retries):
        _, alive = psutil.wait_procs(alive, 1.0)
        if not alive:
            break
    if alive:
        error_msg = ''
        for child in alive:
            error_msg += '{}\n'.format(
                str(
                    child.as_dict(
                        attrs=['ppid', 'pid', 'name', 'status', 'cmdline'])))

        error_msg = ("The following processes didn't die after the shutdown "
                     'of run_experiment:\n') + error_msg
        error_msg += ('This is a sign of an unclean shutdown. Please reopen '
                      'the following issue\nwith a detailed description '
                      'of how the error was produced:\n')
        error_msg += ('https://github.com/rlworkgroup/garage/issues/120')
        print(error_msg)


def log_parameters(log_file, args):
    """Log parameters to file."""
    log_params = {}
    for param_name, param_value in args.__dict__.items():
        log_params[param_name] = param_value
    if args.args_data is not None:
        log_params['json_args'] = dict()
    pathlib.Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(log_params, f, indent=2, sort_keys=True, cls=LogEncoder)


def dump_variant(log_file, variant_data):
    """Dump the variant file."""
    pathlib.Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(variant_data, f, indent=2, sort_keys=True, cls=LogEncoder)


class LogEncoder(json.JSONEncoder):
    """Encoder to be used as cls in json.dump."""

    def default(self, o):  # pylint: disable=method-hidden
        """Perform JSON encoding."""
        if isinstance(o, type):
            return {'$class': o.__module__ + '.' + o.__name__}
        elif isinstance(o, enum.Enum):
            return {
                '$enum':
                o.__module__ + '.' + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {'$function': o.__module__ + '.' + o.__name__}
        return json.JSONEncoder.default(self, o)


if __name__ == '__main__':
    run_experiment(sys.argv)
