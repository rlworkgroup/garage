"""helper functions for benchmarks."""
import csv
import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def create_json(csvs, trials, seeds, xs, ys, factors, names):
    """Convert multiple algorithms csvs outputs to json format.

    Args:
        csvs (list[list]): A list of list of csvs which contains all csv files
            for each algorithms in the task.
        trials (int): Number of trials in the task.
        seeds (list[int]): A list of positive integers which is used in the
            algorithms
        xs (list[string]): A list of X column names of algorithms csv.
        ys (list[string]): A list of Y column names of algorithms csv.
        factors (list[int]): A list of factor value for each algorithms
        names (list[string]): A list of name of each algorithms

    Returns:
        dict: a dictionary(json) whose values should contain time_steps
            (x-value) and return values (y-value) for each algorithms for each
            trials

    """
    task_result = {}
    for trial in range(trials):
        trial_seed = 'trial_%d' % (trial + 1)
        task_result['seed'] = seeds[trial]
        task_result[trial_seed] = {}

        dfs = (json.loads(pd.read_csv(csv[trial]).to_json()) for csv in csvs)
        task_result[trial_seed] = {
            name: {
                'time_steps': [float(val) * factor for val in df[x].values()],
                'return': df[y]
            }
            for df, x, y, factor, name in zip(dfs, xs, ys, factors, names)
        }
    return task_result


def plot_average_over_trials(csvs, ys, plt_file, env_id, x_label, y_label,
                             names):
    """Plot mean and confidence area of benchmark from csv files of algorithms.

    x-value is step and y-value depends on the parameter ys.
    Calculate mean and std for the y values and draw a line using mean and
    show confidence area using std.

    Step length of every csv data ans ys should be same.

    Args:
        csvs (list[list]): A list of list of csvs which contains all csv files
            for each algorithms in the task.
        ys (list[int]): A list of Y column names of algorithms csv.
        plt_file (string): Path of the plot png file.
        env_id (string): String contains the id of the environment. (for title)
        x_label (string): label for x axis of the plot
        y_label (string): label for y axis of the plot
        names (list[string]): labels for each line in the graph

    """
    assert all(len(x) == len(csvs[0]) for x in csvs)

    for trials, y, name in zip(csvs, ys, names):
        y_vals = np.array([np.array(pd.read_csv(t)[y]) for t in trials])
        y_mean, y_std = y_vals.mean(axis=0), y_vals.std(axis=0)

        # pylint: disable=unsubscriptable-object
        plt.plot(list(range(y_vals.shape[-1])), y_mean, label=name)
        # pylint: disable=unsubscriptable-object
        plt.fill_between(list(range(y_vals.shape[-1])), (y_mean - y_std),
                         (y_mean + y_std),
                         alpha=.1)

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(env_id)

    plt.savefig(plt_file)
    plt.close()


def iterate_experiments(funcname,
                        tasks,
                        seeds,
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
        funcname (str): The experiment function name.
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
    for task in tasks:
        env_id = task['env_id']
        for seed in seeds:
            # This breaks algorithm and implementation name with '_'
            # For example: ppo_garage_tf -> ppo_garage-tf
            i = funcname.find('_')
            s = funcname[:i + 1] + funcname[i + 1:].replace('_', '-')
            name = s + '_' + env_id + '_' + str(seed)

            log_dir = get_log_dir(name)

            if use_tf or funcname.endswith('tf'):
                tf.compat.v1.reset_default_graph()

            yield env_id, seed, log_dir

            csv_to_json(log_dir, xcolumn, xlabel, ycolumn, ylabel)


def get_log_dir(name):
    """Get the log directory given the experiment name.

    Args:
        name (str): The experiment name.

    Returns:
        str: Log directory.

    """
    cwd = pathlib.Path.cwd()
    return str(cwd.joinpath('data', 'local', 'benchmarks', name))


def csv_to_json(log_dir, xcolumn, xlabel, ycolumn, ylabel):
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
