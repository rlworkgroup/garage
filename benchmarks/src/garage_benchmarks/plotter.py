import os
from garage_benchmarks.parameters import Atari10M_ENV_SET
import matplotlib.pyplot as plt
import numpy as np

from helper import _read_csv

seeds = [0, 1, 2]  # potentially change this to match your seed values.
xcolumn = 'TotalEnvSteps'
xlabel = 'Total Environment Steps'
ycolumn = 'Evaluation/100EpRewardMean'
ylabel = '100 Episode Avg Return'

# Mujoco1M envs by default, change if needed.
env_ids = Atari10M_ENV_SET
print(env_ids)

plot_dir = '/home/mishari/plots'  # make sure this dir exists, the script doesn't create it

fig, ax = plt.subplots(2,4)
fig.set_size_inches(16,9)

for idx, env_id in enumerate(env_ids):

    # change these directories to point to your benchmarks
    dir = [
        # '/home/mishari/new_benchmarks/a2c_benchmarks_tf/a2c-garage-tf_' +
        # env_id + '_',
        '/home/mishari/dqn_benchmarks/'
        + env_id.split('NoFrameskip')[0] + '/' + env_id, '/home/mishari/dqn_benchmarks/' + env_id.split('NoFrameskip')[0] + '/baselines'
    ]
    labels = ['dqn_garage', 'dqn_baselines']

    _plot = {}

    if _plot is not None and env_id not in _plot:
        _plot[env_id] = {'xlabel': xlabel, 'ylabel': ylabel}

    row = 0
    col=idx
    if idx > 3:
        row = 1
        col = idx-4
    minlen = 1e12
    for label, subdir in zip(labels[:-1], dir[:-1]):
        task_ys = []
        for seed in seeds:
            if seed == 0:
                xs, ys = _read_csv(subdir, xcolumn, ycolumn)
            else:
                xs, ys = _read_csv(subdir + '_' + str(seed), xcolumn, ycolumn)
            if len(ys) < minlen:
                minlen = len(ys)
            task_ys.append(ys)

        for i in range(len(task_ys)):
            task_ys[i] = np.array(task_ys[i])
            if len(task_ys[i]) != minlen:
                task_ys[i] = task_ys[i][:minlen]
        ys_mean = np.array(task_ys).mean(axis=0)
        ys_std = np.array(task_ys).std(axis=0)
        xs = xs[:minlen]
        ax[row, col].plot(xs, ys_mean, label=label)
        ax[row, col].fill_between(xs, (ys_mean - ys_std), (ys_mean + ys_std), alpha=.1)

    baselines_seeds = [0, 1, 2]
    garage_min = minlen

    task_ys = []
    minlen = 1e12
    for seed in baselines_seeds:
        xs, ys = _read_csv(dir[-1] + '/seed-' + str(seed), 'steps',
                           'mean 100 episode reward')
        if len(ys) < minlen:
            minlen = len(ys)
        task_ys.append(ys)

    for i in range(len(task_ys)):
        task_ys[i] = np.array(task_ys[i])
        if len(task_ys[i]) != minlen:
            task_ys[i] = task_ys[i][:minlen]
    xs = xs[:minlen]
    garage_minstep = xs[minlen-1]

    ys_mean = np.array(task_ys).mean(axis=0)
    ys_std = np.array(task_ys).std(axis=0)

    ax[row, col].plot(xs, ys_mean, label=labels[-1])
    ax[row, col].fill_between(xs, (ys_mean - ys_std), (ys_mean + ys_std), alpha=.1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax[row, col].legend()
    ax[row, col].set_title(env_id)
    # if 'Breakout' in env_id:
    #     ax[row, col].set_xlim([0, 5e6])
    # else:
    #     ax[row, col].set_xlim([0, xs[minlen-1]])
    ax[row, col].set_xlim([0, max(xs[minlen-1], garage_minstep)])
plt.savefig(plot_dir + '/' + 'subplot')
print(plot_dir + '/' + env_id)
