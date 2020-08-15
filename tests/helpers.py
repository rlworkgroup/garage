"""helper functions for tests and benchmarks."""
import cloudpickle
import numpy as np
import pytest

from garage import Environment

from tests.quirks import KNOWN_GYM_RENDER_NOT_IMPLEMENTED


def step_env(env, n=10, visualize=True):
    """Step env helper.

    Args:
        env (Environment): Input environment.
        n (int): Steps.
        visualize (bool): Whether visualize the environment.

    """
    env.reset()
    if visualize and issubclass(type(env), Environment):
        env.visualize()
    for _ in range(n):
        print('itr:', _)
        es = env.step(env.action_space.sample())
        if es.last:
            break


def step_env_with_gym_quirks(env,
                             spec,
                             n=10,
                             visualize=True,
                             serialize_env=False):
    """Step env helper.

    Args:
        env (Environment): Input environment.
        spec (EnvSpec): The environment specification.
        n (int): Steps.
        visualize (bool): Whether to visualize the environment.
        serialize_env (bool): Whether to serialize the environment.

    """
    if serialize_env:
        # Roundtrip serialization
        round_trip = cloudpickle.loads(cloudpickle.dumps(env))
        assert round_trip.spec == env.spec
        env = round_trip

    env.reset()
    for _ in range(n):
        es = env.step(env.action_space.sample())
        if visualize:
            if spec.id not in KNOWN_GYM_RENDER_NOT_IMPLEMENTED:
                env.visualize()
            else:
                with pytest.raises(NotImplementedError):
                    env.visualize()
        if es.last:
            break

    env.close()


def convolve(_input, filter_weights, filter_bias, strides, filters,
             in_channels, hidden_nonlinearity):
    """Helper function for performing convolution.

    Args:
        _input (tf.Tensor): Input tf.Tensor to the CNN.
        filter_weights (tuple(tf.Tensor)): The weights of the filters.
        filter_bias (tuple(tf.Tensor)): The bias of the filters.
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        filters (Tuple[Tuple[int, Tuple[int, int]], ...]): Number and dimension
            of filters. For example, ((3, (3, 5)), (32, (3, 3))) means there
            are two convolutional layers. The filter for the first layer have 3
            channels and its shape is (3 x 5), while the filter for the second
            layer have 32 channels and its shape is (3 x 3).
        in_channels (tuple[int]): The number of input channels.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.

    Return:
        tf.Tensor: The output of the convolution.

    """
    batch_size = _input.shape[0]
    in_width = _input.shape[1]
    in_height = _input.shape[2]

    for filter_iter, in_shape, filter_weight, _filter_bias, stride in zip(
            filters, in_channels, filter_weights, filter_bias, strides):
        filter_width = filter_iter[1][1]
        filter_height = filter_iter[1][0]
        out_width = int((in_width - filter_width) / stride) + 1
        out_height = int((in_height - filter_height) / stride) + 1
        flatten_filter_size = filter_width * filter_height * in_shape
        reshape_filter = filter_weight.reshape(flatten_filter_size, -1)
        image_vector = np.empty(
            (batch_size, out_width, out_height, flatten_filter_size))
        for batch in range(batch_size):
            for w in range(out_width):
                for h in range(out_height):
                    image_vector[batch][w][h] = _construct_image_vector(
                        _input, batch, w, h, filter_width, filter_height,
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
    """Helper function for performing feedforward of a lstm cell.

    Args:
        input_val (tf.Tensor): Input placeholder.
        num_units (int): Hidden dimension for LSTM cell.
        step_hidden (tf.Tensor): Place holder for step hidden state.
        step_cell (tf.Tensor): Place holder for step cell state.
        nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        w_x_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        gate_nonlinearity (callable): Activation function for recurrent
            layers. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        w_h_init (callable): Initializer function for the weight
            of recurrent layer(s). The function should return a
            tf.Tensor.
        forget_bias (float): Bias to be added to the forget gate at
            initialization. It's used to reduce the scale of forgetting at the
            beginning of the training.

    Returns:
        tf.Tensor: Final hidden state after feedforward.
        tf.Tensor: Final cell state after feedforward.

    Note:
        Incoming gate:    i(t) = f_i(x(t) @ W_xi + h(t-1) @ W_hi +
                                 w_ci * c(t-1) + b_i)
        Forget gate:      f(t) = f_f(x(t) @ W_xf + h(t-1) @ W_hf +
                                 w_cf * c(t-1) + b_f)
        Cell gate:        c(t) = f(t) * c(t - 1) + i(t) * f_c(x(t) @ W_xc +
                                 h(t-1) @ W_hc + b_c)
        Out gate:         o(t) = f_o(x(t) @ W_xo + h(t-1) W_ho +
                                 w_co * c(t) + b_o)
        New hidden state: h(t) = o(t) * f_h(c(t))
        Incoming, forget, cell, and out vectors must have the same
        dimension as the hidden state.

    """

    def f(x):
        """Linear function.

        Args:
            x (float): Input variable.

        Returns:
            float: Ouput variable.

        """
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
    """Helper function for performing feedforward of a GRU cell.

    Args:
        input_val (tf.Tensor): Input placeholder.
        num_units (int): Hidden dimension for GRU cell.
        step_hidden (tf.Tensor): Place holder for step hidden state.
        nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        w_x_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        gate_nonlinearity (callable): Activation function for recurrent
            layers. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        w_h_init (callable): Initializer function for the weight
            of recurrent layer(s). The function should return a
            tf.Tensor.
        forget_bias (float): Bias to be added to the forget gate at
            initialization. It's used to reduce the scale of forgetting at the
            beginning of the training.

    Returns:
        tf.Tensor: Final hidden state after feedforward.

    Note:
        Reset gate:        r(t) = f_r(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
        Update gate:       u(t) = f_u(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
        Cell gate:         c(t) = f_c(x(t) @ W_xc + r(t) *
                                  (h(t-1) @ W_hc) + b_c)
        New hidden state:  h(t) = u_t * h(t-1) + (1 - u(t)) * c(t)
        The reset, update, and cell vectors must have the same dimension
        as the hidden state.

    """

    def f(x):
        """Linear function.

        Args:
            x (float): Input variable.

        Returns:
            float: Ouput variable.

        """
        return x

    del forget_bias
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
    """Get sliding window of input image.

    Args:
        _input (tf.Tensor): Input tf.Tensor to the CNN.
        batch (int): Batch index.
        w (int): Width index.
        h (int): Height index.
        filter_width (int): Width of the filter.
        filter_height (int): Height of the filter.
        in_shape (int): The number of input channels.

    Return:
        np.array: The output of the sliding window.

    """
    sw = np.empty((filter_width, filter_height, in_shape))
    for dw in range(filter_width):
        for dh in range(filter_height):
            for in_c in range(in_shape):
                sw[dw][dh][in_c] = _input[batch][w + dw][h + dh][in_c]
    return sw.flatten()


def max_pooling(_input, pool_shape, pool_stride, padding='VALID'):
    """Helper function for performing max pooling.

    Args:
        _input (tf.Tensor): Input tf.Tensor to the CNN.
        pool_shape (int): Dimension of the pooling layer.
        pool_stride (int): The stride of the pooling layer.
        padding (str): The type of padding algorithm to use, either 'SAME'
            or 'VALID'.

    Return:
        tf.Tensor: The output tf.Tensor after max pooling.

    """
    batch_size = _input.shape[0]
    if padding == 'VALID':
        height_size = int((_input.shape[1] - pool_shape) / pool_stride) + 1
        width_size = int((_input.shape[2] - pool_shape) / pool_stride) + 1
    else:
        height_size = int((_input.shape[1] + pool_stride - 1) / pool_stride)
        width_size = int((_input.shape[2] + pool_stride - 1) / pool_stride)

    # max pooling
    results = np.empty((batch_size, height_size, width_size, _input.shape[3]))
    for b in range(batch_size):
        for i in range(0, results.shape[1]):
            for j in range(0, results.shape[2]):
                for k in range(_input.shape[3]):
                    row = i * pool_shape
                    col = j * pool_shape
                    results[b][i][j][k] = np.max(_input[b,
                                                        row:row + pool_shape,
                                                        col:col +  # noqa: W504
                                                        pool_shape, k])

    return results
