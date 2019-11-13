"""helper functions for tests and benchmarks."""
import json
import os
import os.path as osp
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from tests.quirks import KNOWN_GYM_RENDER_NOT_IMPLEMENTED


def step_env(env, n=10, render=True):
    """Step env helper."""
    env.reset()
    for _ in range(n):
        _, _, done, _ = env.step(env.action_space.sample())
        if render:
            env.render()
        if done:
            break


def step_env_with_gym_quirks(env, spec, n=10, render=True,
                             serialize_env=False):
    """Step env gym helper."""
    if serialize_env:
        # Roundtrip serialization
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip.env.spec == env.env.spec
        env = round_trip

    env.reset()
    for _ in range(n):
        _, _, done, _ = env.step(env.action_space.sample())
        if render:
            if spec.id not in KNOWN_GYM_RENDER_NOT_IMPLEMENTED:
                env.render()
            else:
                with pytest.raises(NotImplementedError):
                    env.render()
        if done:
            break

    env.close()

    if serialize_env:
        # Roundtrip serialization
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip.env.spec == env.env.spec


def convolve(_input, filter_weights, filter_bias, strides, filter_sizes,
             in_channels, hidden_nonlinearity):
    """Convolve."""
    # in_width = self.input_width
    # in_height = self.input_height

    batch_size = _input.shape[0]
    in_width = _input.shape[1]
    in_height = _input.shape[2]

    for filter_size, in_shape, filter_weight, _filter_bias, stride in zip(
            filter_sizes, in_channels, filter_weights, filter_bias, strides):
        out_width = int((in_width - filter_size) / stride) + 1
        out_height = int((in_height - filter_size) / stride) + 1
        flatten_filter_size = filter_size * filter_size * in_shape
        reshape_filter = filter_weight.reshape(flatten_filter_size, -1)
        image_vector = np.empty(
            (batch_size, out_width, out_height, flatten_filter_size))
        for batch in range(batch_size):
            for w in range(out_width):
                for h in range(out_height):
                    image_vector[batch][w][h] = _construct_image_vector(
                        _input, batch, w, h, filter_size, filter_size,
                        in_shape)

        _input = np.dot(image_vector, reshape_filter) + _filter_bias
        _input = hidden_nonlinearity(_input).eval()

        in_width = out_width
        in_height = out_height

    return _input


def recurrent_step_lstm(input_val,
                        num_units,
                        step_hidden,
                        step_cell,
                        w_x_init,
                        w_h_init,
                        b_init,
                        nonlinearity,
                        gate_nonlinearity,
                        forget_bias=1.0):
    """
    A LSTM unit implements the following update mechanism:

    Incoming gate:    i(t) = f_i(x(t) @ W_xi + h(t-1) @ W_hi +
                                 w_ci * c(t-1) + b_i)
    Forget gate:      f(t) = f_f(x(t) @ W_xf + h(t-1) @ W_hf +
                                 w_cf * c(t-1) + b_f)
    Cell gate:        c(t) = f(t) * c(t - 1) + i(t) * f_c(x(t) @ W_xc +
                             h(t-1) @ W_hc + b_c)
    Out gate:         o(t) = f_o(x(t) @ W_xo + h(t-1) W_ho + w_co * c(t) + b_o)
    New hidden state: h(t) = o(t) * f_h(c(t))

    Note that the incoming, forget, cell, and out vectors must have the same
    dimension as the hidden state.
    """

    def f(x):
        return x

    if nonlinearity is None:
        nonlinearity = f
    if gate_nonlinearity is None:
        gate_nonlinearity = f

    input_dim = np.prod(input_val.shape[1:])
    # Weights for the input gate
    w_xi = np.full((input_dim, num_units), w_x_init)
    w_hi = np.full((num_units, num_units), w_h_init)
    b_i = np.full((num_units, ), b_init)
    # Weights for the forget gate
    w_xf = np.full((input_dim, num_units), w_x_init)
    w_hf = np.full((num_units, num_units), w_h_init)
    b_f = np.full((num_units, ), b_init)
    # Weights for the cell gate
    w_xc = np.full((input_dim, num_units), w_x_init)
    w_hc = np.full((num_units, num_units), w_h_init)
    b_c = np.full((num_units, ), b_init)
    # Weights for the out gate
    w_xo = np.full((input_dim, num_units), w_x_init)
    w_ho = np.full((num_units, num_units), w_h_init)
    b_o = np.full((num_units, ), b_init)

    w_x_ifco = np.concatenate([w_xi, w_xf, w_xc, w_xo], axis=1)
    w_h_ifco = np.concatenate([w_hi, w_hf, w_hc, w_ho], axis=1)

    x_ifco = np.matmul(input_val, w_x_ifco)
    h_ifco = np.matmul(step_hidden, w_h_ifco)

    x_i = x_ifco[:, :num_units]
    x_f = x_ifco[:, num_units:num_units * 2]
    x_c = x_ifco[:, num_units * 2:num_units * 3]
    x_o = x_ifco[:, num_units * 3:num_units * 4]

    h_i = h_ifco[:, :num_units]
    h_f = h_ifco[:, num_units:num_units * 2]
    h_c = h_ifco[:, num_units * 2:num_units * 3]
    h_o = h_ifco[:, num_units * 3:num_units * 4]

    i = gate_nonlinearity(x_i + h_i + b_i)
    f = gate_nonlinearity(x_f + h_f + b_f + forget_bias)
    o = gate_nonlinearity(x_o + h_o + b_o)

    c = f * step_cell + i * nonlinearity(x_c + h_c + b_c)
    h = o * nonlinearity(c)
    return h, c


def recurrent_step_gru(input_val,
                       num_units,
                       step_hidden,
                       w_x_init,
                       w_h_init,
                       b_init,
                       nonlinearity,
                       gate_nonlinearity,
                       forget_bias=1.0):
    """
    A GRU unit implements the following update mechanism:

    Reset gate:        r(t) = f_r(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
    Update gate:       u(t) = f_u(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
    Cell gate:         c(t) = f_c(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
    New hidden state:  h(t) = u_t * h(t-1) + (1 - u(t)) * c(t)

    Note that the reset, update, and cell vectors must have the same dimension
    as the hidden state.
    """

    def f(x):
        return x

    if nonlinearity is None:
        nonlinearity = f
    if gate_nonlinearity is None:
        gate_nonlinearity = f

    input_dim = np.prod(input_val.shape[1:])
    # Weights for the update gate
    w_xz = np.full((input_dim, num_units), w_x_init)
    w_hz = np.full((num_units, num_units), w_h_init)
    b_z = np.full((num_units, ), b_init)
    # Weights for the reset gate
    w_xr = np.full((input_dim, num_units), w_x_init)
    w_hr = np.full((num_units, num_units), w_h_init)
    b_r = np.full((num_units, ), b_init)
    # Weights for the hidden gate
    w_xh = np.full((input_dim, num_units), w_x_init)
    w_hh = np.full((num_units, num_units), w_h_init)
    b_h = np.full((num_units, ), b_init)

    w_x_zrh = np.concatenate([w_xz, w_xr, w_xh], axis=1)
    w_h_zrh = np.concatenate([w_hz, w_hr, w_hh], axis=1)

    x_zrh = np.matmul(input_val, w_x_zrh)
    h_zrh = np.matmul(step_hidden, w_h_zrh)

    x_z = x_zrh[:, :num_units]
    x_r = x_zrh[:, num_units:num_units * 2]
    x_h = x_zrh[:, num_units * 2:num_units * 3]

    h_z = h_zrh[:, :num_units]
    h_r = h_zrh[:, num_units:num_units * 2]
    h_h = h_zrh[:, num_units * 2:num_units * 3]

    z = gate_nonlinearity(x_z + h_z + b_z)
    r = gate_nonlinearity(x_r + h_r + b_r)
    hh = nonlinearity(x_h + r * h_h + b_h)
    h = z * step_hidden + (1 - z) * hh

    return h


def _construct_image_vector(_input, batch, w, h, filter_width, filter_height,
                            in_shape):
    """sw is sliding window"""
    sw = np.empty((filter_width, filter_height, in_shape))
    for dw in range(filter_width):
        for dh in range(filter_height):
            for in_c in range(in_shape):
                sw[dw][dh][in_c] = _input[batch][w + dw][h + dh][in_c]
    return sw.flatten()


def max_pooling(_input, pool_shape, pool_stride):
    """Max pooling."""
    batch_size = _input.shape[0]

    # max pooling
    results = np.empty((batch_size, int(_input.shape[1] / pool_shape),
                        int(_input.shape[2] / pool_shape), _input.shape[3]))
    for b in range(batch_size):
        for i, row in enumerate(range(0, _input.shape[1], pool_stride)):
            for j, col in enumerate(range(0, _input.shape[2], pool_stride)):
                for k in range(_input.shape[3]):
                    results[b][i][j][k] = np.max(
                        _input[b, col:col + pool_shape, row:row +  # noqa: W504
                               pool_shape, k])

    return results


def write_file(result_json, algo):
    """Create new progress.json or append to existing one."""
    latest_dir = './latest_results'
    latest_result = latest_dir + '/progress.json'
    res = {}
    if osp.exists(latest_result):
        res = json.loads(open(latest_result, 'r').read())
    elif not osp.exists(latest_dir):
        os.makedirs(latest_dir)
    res[algo] = result_json
    result_file = open(latest_result, 'w')
    result_file.write(json.dumps(res))


def create_json(b_csvs, g_csvs, trails, seeds, b_x, b_y, g_x, g_y, factor_g,
                factor_b):
    """Convert garage and benchmark csv outputs to json format."""
    task_result = {}
    for trail in range(trails):
        g_res, b_res = {}, {}
        trail_seed = 'trail_%d' % (trail + 1)
        task_result['seed'] = seeds[trail]
        task_result[trail_seed] = {}
        df_g = json.loads(pd.read_csv(g_csvs[trail]).to_json())
        df_b = json.loads(pd.read_csv(b_csvs[trail]).to_json())

        g_res['time_steps'] = list(
            map(lambda x: float(x) * factor_g, df_g[g_x].values()))
        g_res['return'] = df_g[g_y]

        b_res['time_steps'] = list(
            map(lambda x: float(x) * factor_b, df_b[b_x].values()))
        b_res['return'] = df_b[b_y]

        task_result[trail_seed]['garage'] = g_res
        task_result[trail_seed]['baselines'] = b_res
    return task_result


def relplot(g_csvs, b_csvs, g_x, g_y, g_z, b_x, b_y, b_z, trials, seeds,
            plt_file, env_id, x_label, y_label):
    """
    Plot benchmark from csv files of garage from multiple trials using Seaborn.

    :param g_csvs: A list contains all csv files in the task.
    :param b_csvs: A list contains all csv files in the task. Pass
        empty list or None if only plotting garage.
    :param g_x: X column names of garage csv.
    :param g_y: Y column names of garage csv.
    :param b_x: X column names of baselines csv.
    :param b_y: Y column names of baselines csv.
    :param trials: Number of trials in the task.
    :param seeds: A list contains all the seeds in the task.
    :param plt_file: Path of the plot png file.
    :param env_id: String contains the id of the environment.
    :return:
    """
    df_g = [pd.read_csv(g) for g in g_csvs]
    df_gs = pd.concat(df_g, axis=0)
    df_gs['Type'] = g_z
    data = df_gs

    if b_csvs:
        assert len(b_csvs) == len(g_csvs)
        df_b = [pd.read_csv(b) for b in b_csvs]
        df_bs = pd.concat(df_b, axis=0)
        df_bs['Type'] = b_z
        df_bs = df_bs.rename(columns={b_x: g_x, b_y: g_y})
        data = pd.concat([df_gs, df_bs])

    ax = sns.relplot(x=g_x, y=g_y, hue='Type', kind='line', data=data)
    ax.axes.flatten()[0].set_title(env_id)

    plt.savefig(plt_file)

    plt.close()


def plot(g_csvs, b_csvs, g_x, g_y, g_z, b_x, b_y, b_z, trials, seeds, plt_file,
         env_id, x_label, y_label):
    """
    Plot benchmark from csv files of garage and baselines.

    :param b_csvs: A list contains all csv files in the task.
    :param g_csvs: A list contains all csv files in the task.
    :param g_x: X column names of garage csv.
    :param g_y: Y column names of garage csv.
    :param b_x: X column names of baselines csv.
    :param b_y: Y column names of baselines csv.
    :param trials: Number of trials in the task.
    :param seeds: A list contains all the seeds in the task.
    :param plt_file: Path of the plot png file.
    :param env_id: String contains the id of the environment.
    :return:
    """
    assert len(b_csvs) == len(g_csvs)
    for trial in range(trials):
        seed = seeds[trial]

        df_g = pd.read_csv(g_csvs[trial])
        df_b = pd.read_csv(b_csvs[trial])

        plt.plot(df_g[g_x],
                 df_g[g_y],
                 label='%s_trial%d_seed%d' % (g_z, trial + 1, seed))
        plt.plot(df_b[b_x],
                 df_b[b_y],
                 label='%s_trial%d_seed%d' % (b_z, trial + 1, seed))

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(env_id)

    plt.savefig(plt_file)
    plt.close()
