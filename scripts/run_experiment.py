import argparse
import ast
import base64
import datetime
import os
import os.path as osp
import pickle as pickle
import sys
sys.path.append(".")
import uuid

import dateutil.tz
import joblib

from garage import config
from garage.misc import gcs_utils
from garage.misc.ext import is_iterable
from garage.misc.ext import set_seed
from garage.misc.instrument import concretize
import garage.misc.logger as logger
import garage.plotter
import garage.tf.plotter


def run_experiment(argv):
    default_log_dir = config.LOG_DIR
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
        help=("Number of parallel workers to perform rollouts. "
              "0 => don't start any workers"))
    parser.add_argument(
        '--exp_name',
        type=str,
        default=default_exp_name,
        help='Name of the experiment.')
    parser.add_argument(
        '--log_dir',
        type=str,
        default=None,
        help='Path to save the log and iteration snapshot.')
    parser.add_argument(
        '--snapshot_mode',
        type=str,
        default='all',
        help='Mode to save the snapshot. Can be either "all" '
        '(all iterations will be saved), "last" (only '
        'the last iteration will be saved), "gap" (every'
        '`snapshot_gap` iterations are saved), or "none" '
        '(do not save snapshots)')
    parser.add_argument(
        '--snapshot_gap',
        type=int,
        default=1,
        help='Gap between snapshot iterations.')
    parser.add_argument(
        '--tabular_log_file',
        type=str,
        default='progress.csv',
        help='Name of the tabular log file (in csv).')
    parser.add_argument(
        '--text_log_file',
        type=str,
        default='debug.log',
        help='Name of the text log file (in pure text).')
    parser.add_argument(
        '--tensorboard_step_key',
        type=str,
        default=None,
        help=("Name of the step key in tensorboard_summary."))
    parser.add_argument(
        '--params_log_file',
        type=str,
        default='params.json',
        help='Name of the parameter log file (in json).')
    parser.add_argument(
        '--variant_log_file',
        type=str,
        default='variant.json',
        help='Name of the variant log file (in json).')
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Name of the pickle file to resume experiment from.')
    parser.add_argument(
        '--plot',
        type=ast.literal_eval,
        default=False,
        help='Whether to plot the iteration results')
    parser.add_argument(
        '--log_tabular_only',
        type=ast.literal_eval,
        default=False,
        help='Print only the tabular log information (in a horizontal format)')
    parser.add_argument('--seed', type=int, help='Random seed for numpy')
    parser.add_argument(
        '--args_data', type=str, help='Pickled data for stub objects')
    parser.add_argument(
        '--variant_data',
        type=str,
        help='Pickled data for variant configuration')
    parser.add_argument(
        '--use_cloudpickle', type=ast.literal_eval, default=False)
    parser.add_argument(
        '--save_in_gcs_bucket',
        type=bool,
        default=False,
        help='If True, the logs and checkpoints of the experiment are saved ' \
        'in the bucket set in config_personal.py')

    args = parser.parse_args(argv[1:])

    assert (os.environ.get("JOBLIB_START_METHOD", None) == "forkserver")

    if args.save_in_gcs_bucket:
        gcs_utils.check_gcs_config()

    if args.seed is not None:
        set_seed(args.seed)

    if args.n_parallel > 0:
        from garage.sampler import parallel_sampler
        parallel_sampler.initialize(n_parallel=args.n_parallel)
        if args.seed is not None:
            parallel_sampler.set_seed(args.seed)

    if not args.plot:
        garage.plotter.Plotter.disable()
        garage.tf.plotter.Plotter.disable()

    if args.log_dir is None:
        log_dir = osp.join(default_log_dir, args.exp_name)
    else:
        log_dir = args.log_dir
    tabular_log_file = osp.join(log_dir, args.tabular_log_file)
    text_log_file = osp.join(log_dir, args.text_log_file)
    params_log_file = osp.join(log_dir, args.params_log_file)

    if args.variant_data is not None:
        variant_data = pickle.loads(base64.b64decode(args.variant_data))
        variant_log_file = osp.join(log_dir, args.variant_log_file)
        logger.log_variant(variant_log_file, variant_data)
    else:
        variant_data = None

    if not args.use_cloudpickle:
        logger.log_parameters_lite(params_log_file, args)

    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    logger.set_tensorboard_dir(log_dir)
    prev_snapshot_dir = logger.get_snapshot_dir()
    prev_mode = logger.get_snapshot_mode()
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(args.snapshot_mode)
    logger.set_snapshot_gap(args.snapshot_gap)
    logger.set_log_tabular_only(args.log_tabular_only)
    logger.set_tensorboard_step_key(args.tensorboard_step_key)
    logger.push_prefix("[%s] " % args.exp_name)

    if args.resume_from is not None:
        data = joblib.load(args.resume_from)
        assert 'algo' in data
        algo = data['algo']
        algo.train()
    else:
        # read from stdin
        if args.use_cloudpickle:
            import cloudpickle
            method_call = cloudpickle.loads(base64.b64decode(args.args_data))
            try:
                method_call(variant_data)
            except BaseException:
                if args.n_parallel > 0:
                    parallel_sampler.terminate()
                raise
        else:
            data = pickle.loads(base64.b64decode(args.args_data))
            maybe_iter = concretize(data)
            if is_iterable(maybe_iter):
                for _ in maybe_iter:
                    pass

    logger.set_snapshot_mode(prev_mode)
    logger.set_snapshot_dir(prev_snapshot_dir)
    logger.remove_tabular_output(tabular_log_file)
    logger.remove_text_output(text_log_file)
    logger.pop_prefix()

    if args.save_in_gcs_bucket:
        gcs_utils.upload_to_gcs(
            log_dir, path_in_bucket=config.GCS_PATH_IN_BUCKET + args.exp_name)


if __name__ == "__main__":
    run_experiment(sys.argv)
