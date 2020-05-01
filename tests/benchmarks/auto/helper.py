"""Helper functions for automatic benchmarks."""
import csv
import json
import os
import pathlib
import random

from baselines.bench import benchmarks
import tensorflow as tf


def iterate_experiments(func,
                        tasks=None,
                        seeds=None,
                        use_tf=False,
                        xcolumn='TotalEnvSteps',
                        xlabel='Total Environment Steps',
                        ycolumn='Evaluation/AverageReturn',
                        ylabel='Average Return'):
    """Iterator to iterate experiments.

    Also it saves csv to JSON format preparing for automatic benchmarking.

    Args:
        tasks (list[dict]): List of running tasks.
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

    Yields:
        str: The next environment id to construct the env.
        int: The next seed value.
        str: The next experiment's log directory.

    """
    funcname = func.__name__
    if tasks is None:
        tasks = benchmarks.get_benchmark('Mujoco1M')['tasks']
    if seeds is None:
        seeds = random.sample(range(100), 4)

    for task in tasks:
        env_id = task['env_id']
        for seed in seeds:
            # This breaks algorithm and implementation name with '_'
            # For example: ppo_garage_tf -> ppo_garage-tf
            i = funcname.find('_')
            s = funcname[:i + 1] + funcname[i + 1:].replace('_', '-')
            name = s + '_' + env_id + '_' + str(seed)

            log_dir = _get_log_dir(name)

            if use_tf or funcname.endswith('tf'):
                tf.compat.v1.reset_default_graph()

            yield env_id, seed, log_dir

            _csv_to_json(log_dir, xcolumn, xlabel, ycolumn, ylabel)


def _get_log_dir(name):
    """Get the log directory given the experiment name.

    Args:
        name (str): The experiment name.

    Returns:
        str: Log directory.

    """
    cwd = pathlib.Path.cwd()
    return str(cwd.joinpath('data', 'local', 'benchmarks', name))


def _csv_to_json(log_dir, xcolumn, xlabel, ycolumn, ylabel):
    """Save selected csv column to JSON preparing for automatic benchmarking.

    Args:
        log_dir (str): Log directory for csv file.
        xcolumn (str): Which column should be the JSON x axis.
        xlabel (str): Label name for x axis.
        ycolumn (str): Which column should be the JSON y axis.
        ylabel (str): Label name for y axis.

    """
    with open(os.path.join(log_dir, 'progress.csv'), 'r') as csv_file:
        xs, ys = [], []
        for row in csv.DictReader(csv_file):
            xs.append(row[xcolumn])
            ys.append(row[ycolumn])

    with open(os.path.join(log_dir, 'progress.json'), 'w') as json_file:
        json.dump(dict(x=xs, y=ys, xlabel=xlabel, ylabel=ylabel), json_file)
