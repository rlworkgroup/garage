"""Helper functions for benchmarks.

A decorator and a function are provided for benchmarks. They essentially
take care log directory configuration and result plotting.

The decorator can be with or without parameters to specify options for
plotting and JSON exporting used for automatic benchmarking.

If plot is enabled, iterate_experiments() can be called with different
experiments function, each of which can have its specified env_ids. In
the end, figures will be generated per env_id.

Example:
    @benchmark
    def ppo_benchmarks():
        iterate_experiments(ppo_garage_tf)
        iterate_experiments(ppo_garage_pytorch)

    @benchmark(plot=False)
    def my_benchmarks():
        iterate_experiments(trpo_garage_pytorch, env_ids=env_ids)
        iterate_experiments(vpg_garage_pytorch, seeds=seeds)

Attributes:
    _plot (dict): A dictionary used for plotting to figures
        grouped by tasks. If no need to plot, _plot will be None.
    _log_dir (str): Log directory for running the benchmarking.
    _auto (bool): Global variable used to control whether to export
        JSON files for automatic benchmarking.

"""
import csv
import functools
import json
import os
import pathlib
import random
import shutil

from baselines.bench import benchmarks
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

_plot = None
_log_dir = None
_auto = False


def benchmark(exec_func=None, *, plot=True, auto=False):
    """Decorator for benchmark function.

    Args:
        exec_func (func): The experiment function.
        plot (bool): Whether the result of this run needs to be plotted.
            PNG files will be generated in sub folder /plot.
        auto (auto): Whether this is automatic benchmarking. JSON files
            will be generated in sub folder /auto.

    Returns:
        func: The wrapper function.

    """
    if exec_func is None:
        return functools.partial(benchmark, plot=plot, auto=auto)

    @functools.wraps(exec_func)
    def wrapper_func():
        """The wrapper function."""
        # pylint: disable=global-statement
        global _plot, _log_dir, _auto
        _plot = {} if plot else None
        plt.close('all')

        _log_dir = _get_log_dir(exec_func.__name__)
        if os.path.exists(_log_dir):
            shutil.rmtree(_log_dir)

        if auto:
            _auto = auto
            auto_dir = os.path.join(_log_dir, 'auto')
            os.makedirs(auto_dir)

        exec_func()

        if plot:
            plot_dir = os.path.join(_log_dir, 'plot')
            os.makedirs(plot_dir)
            for env_id in _plot:
                plt.figure(env_id)
                plt.legend()
                plt.xlabel(_plot[env_id]['xlabel'])
                plt.ylabel(_plot[env_id]['ylabel'])
                plt.title(env_id)
                plt.savefig(plot_dir + '/' + env_id)

    return wrapper_func


def iterate_experiments(func,
                        env_ids=None,
                        seeds=None,
                        use_tf=False,
                        xcolumn='TotalEnvSteps',
                        xlabel='Total Environment Steps',
                        ycolumn='Evaluation/AverageReturn',
                        ylabel='Average Return'):
    """Iterate experiments for benchmarking over env_ids and seeds.

    Args:
        env_ids (list[str]): List of environment ids.
        seeds (list[int]): List of seeds.
        func (func): The experiment function.
        use_tf (bool): Whether TF is used. When True, a TF Graph context
            is used for each experiment. The default is set to False.
            However, if the function name ends with 'tf', it automatically
            create a TF Graph context.
        xcolumn (str): Which column should be the JSON x axis.
        xlabel (str): Label name for x axis.
        ycolumn (str): Which column should be the JSON y axis.
        ylabel (str): Label name for y axis.

    """
    func_name = func.__name__.replace('_', '-')
    if env_ids is None:
        tasks = benchmarks.get_benchmark('Mujoco1M')['tasks']
        env_ids = [task['env_id'] for task in tasks]
    if seeds is None:
        seeds = random.sample(range(100), 4)

    for env_id in env_ids:
        task_ys = []

        if _plot is not None and env_id not in _plot:
            _plot[env_id] = {'xlabel': xlabel, 'ylabel': ylabel}

        for seed in seeds:
            exp_name = func_name + '_' + env_id + '_' + str(seed)
            sub_log_dir = os.path.join(_log_dir, exp_name)

            if use_tf or func_name.endswith('tf'):
                tf.compat.v1.reset_default_graph()

            func(dict(log_dir=sub_log_dir), env_id=env_id, seed=seed)

            if _plot is not None or _auto:
                xs, ys = _read_csv(sub_log_dir, xcolumn, ycolumn)
                task_ys.append(ys)

        if _plot is not None or _auto:
            ys_mean = np.array(task_ys).mean(axis=0)
            ys_std = np.array(task_ys).std(axis=0)

            if _plot is not None:
                plt.figure(env_id)

                plt.plot(xs, ys_mean, label=func_name)
                plt.fill_between(xs, (ys_mean - ys_std), (ys_mean + ys_std),
                                 alpha=.1)
            if _auto:
                _export_to_json(env_id + '_' + func_name, xs, xlabel, ys_mean,
                                ylabel, ys_std)


def _get_log_dir(exec_func_name):
    """Get the log directory given the experiment name.

    Args:
        exec_func_name (str): The function name which runs benchmarks.

    Returns:
        str: Log directory.

    """
    cwd = pathlib.Path.cwd()
    return str(cwd.joinpath('data', 'local', 'benchmarks', exec_func_name))


def _read_csv(log_dir, xcolumn, ycolumn):
    """Read csv files and return xs and ys.

    Args:
        log_dir (str): Log directory for csv file.
        xcolumn (str): Which column should be the JSON x axis.
        ycolumn (str): Which column should be the JSON y axis.

    Returns:
        list: List of x axis points.
        list: List of y axis points.

    """
    xs, ys = [], []
    with open(os.path.join(log_dir, 'progress.csv'), 'r') as csv_file:
        for row in csv.DictReader(csv_file):
            xs.append(float(row[xcolumn]))
            ys.append(float(row[ycolumn]))

    return xs, ys


def _export_to_json(json_name, xs, xlabel, ys, ylabel, ys_std):
    """Save selected csv column to JSON preparing for automatic benchmarking.

    Args:
        json_name (str): The JSON file name.
        xs (list): List of x axis points
        xlabel (str): Label name for x axis.
        ys (np.array): List of y axis points
        ylabel (str): Label name for y axis.
        ys_std (np.array): Standard deviation of y asis, used to calculate
            upper and lower boundary for confidence interval.

    """
    json_path = os.path.join(_log_dir, 'auto', json_name + '.json')
    with open(json_path, 'w') as json_file:
        json.dump(
            dict(x=xs,
                 y=ys.tolist(),
                 y_min=(ys - ys_std).tolist(),
                 y_max=(ys + ys_std).tolist(),
                 xlabel=xlabel,
                 ylabel=ylabel), json_file)
