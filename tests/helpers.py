import pickle

import numpy as np

from tests.quirks import KNOWN_GYM_RENDER_NOT_IMPLEMENTED


def step_env(env, n=10, render=True):
    env.reset()
    for _ in range(n):
        _, _, done, _ = env.step(env.action_space.sample())
        if render:
            env.render()
        if done:
            break
    env.close()


def step_env_with_gym_quirks(test_case,
                             env,
                             spec,
                             n=10,
                             render=True,
                             serialize_env=False):
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
                with test_case.assertRaises(NotImplementedError):
                    env.render()
        if done:
            break

    if serialize_env:
        # Roundtrip serialization
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip.env.spec == env.env.spec

    env.close()


def convolve(_input, filter_weights, filter_bias, stride, filter_sizes,
             in_channels, hidden_nonlinearity):
    # in_width = self.input_width
    # in_height = self.input_height

    batch_size = _input.shape[0]
    in_width = _input.shape[1]
    in_height = _input.shape[2]

    for filter_size, in_shape, filter_weight, _filter_bias in zip(
            filter_sizes, in_channels, filter_weights, filter_bias):
        out_width = int((in_width - filter_size) / stride) + 1
        out_height = int((in_height - filter_size) / stride) + 1
        flatten_filter_size = filter_size * filter_size * in_shape
        reshape_filter = filter_weight.reshape(flatten_filter_size, -1)
        image_vector = np.empty((batch_size, out_width, out_height,
                                 flatten_filter_size))
        for batch in range(batch_size):
            for w in range(out_width):
                for h in range(out_height):
                    sliding_window = np.empty((filter_size, filter_size,
                                               in_shape))
                    for dw in range(filter_size):
                        for dh in range(filter_size):
                            for in_c in range(in_shape):
                                sliding_window[dw][dh][in_c] = _input[batch][
                                    w + dw][h + dh][in_c]
                    image_vector[batch][w][h] = sliding_window.flatten()
        _input = np.dot(image_vector, reshape_filter) + _filter_bias
        _input = hidden_nonlinearity(_input).eval()

        in_width = out_width
        in_height = out_height

    return _input


def max_pooling(_input, pool_shape, pool_stride):
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
