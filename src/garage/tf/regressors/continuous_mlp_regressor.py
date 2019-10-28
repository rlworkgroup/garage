"""A regressor based on a MLP model."""
from dowel import tabular
import numpy as np
import tensorflow as tf

from garage.tf.misc import tensor_utils
from garage.tf.models import NormalizedInputMLPModel
from garage.tf.optimizers import LbfgsOptimizer
from garage.tf.regressors import Regressor


class ContinuousMLPRegressor(Regressor):
    """Fits continuously-valued data to an MLP model.

    Args:
        input_shape (tuple[int]): Input shape of the training data.
        output_dim (int): Output dimension of the model.
        name (str): Model name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (Callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (Callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (Callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (Callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (Callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (Callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        optimizer (garage.tf.Optimizer): Optimizer for minimizing the negative
            log-likelihood.
        optimizer_args (dict): Arguments for the optimizer. Default is None,
            which means no arguments.
        normalize_inputs (bool): Bool for normalizing inputs or not.

    """

    def __init__(self,
                 input_shape,
                 output_dim,
                 name='ContinuousMLPRegressor',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 optimizer=None,
                 optimizer_args=None,
                 normalize_inputs=True):
        super().__init__(input_shape, output_dim, name)
        self._normalize_inputs = normalize_inputs

        with tf.compat.v1.variable_scope(self._name, reuse=False) as vs:
            self._variable_scope = vs
            if optimizer_args is None:
                optimizer_args = dict()
            if optimizer is None:
                optimizer = LbfgsOptimizer(**optimizer_args)
            else:
                optimizer = optimizer(**optimizer_args)
            self._optimizer = optimizer

        self.model = NormalizedInputMLPModel(
            input_shape=input_shape,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init)

        self._initialize()

    def _initialize(self):
        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, ) +
                                             self._input_shape)

        with tf.compat.v1.variable_scope(self._name) as vs:
            self._variable_scope = vs
            self.model.build(input_var)
            ys_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                              name='ys',
                                              shape=(None, self._output_dim))

            y_hat = self.model.networks['default'].y_hat
            loss = tf.reduce_mean(tf.square(y_hat - ys_var))

            self._f_predict = tensor_utils.compile_function([input_var], y_hat)
            optimizer_args = dict(
                loss=loss,
                target=self,
                network_outputs=[ys_var],
            )

            optimizer_args['inputs'] = [input_var, ys_var]

            with tf.name_scope('update_opt'):
                self._optimizer.update_opt(**optimizer_args)

    def fit(self, xs, ys):
        """Fit with input data xs and label ys.

        Args:
            xs (numpy.ndarray): Input data.
            ys (numpy.ndarray): Output labels.

        """
        if self._normalize_inputs:
            # recompute normalizing constants for inputs
            self.model.networks['default'].x_mean.load(
                np.mean(xs, axis=0, keepdims=True))
            self.model.networks['default'].x_std.load(
                np.std(xs, axis=0, keepdims=True) + 1e-8)

        inputs = [xs, ys]
        loss_before = self._optimizer.loss(inputs)
        tabular.record('{}/LossBefore'.format(self._name), loss_before)
        self._optimizer.optimize(inputs)
        loss_after = self._optimizer.loss(inputs)
        tabular.record('{}/LossAfter'.format(self._name), loss_after)
        tabular.record('{}/dLoss'.format(self._name), loss_before - loss_after)

    def predict(self, xs):
        """Predict y based on input xs.

        Args:
            xs (numpy.ndarray): Input data.

        Return:
            numpy.ndarray: The predicted ys.

        """
        return self._f_predict(xs)

    def predict_sym(self, xs, name=None):
        """Build a symbolic graph of the model prediction.

        Args:
            xs (tf.Tensor): Input tf.Tensor for the input data.
            name (str): Name of the new graph.

        Return:
            tf.Tensor: Output of the symbolic prediction graph.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            y_hat, _, _ = self.model.build(xs, name=name)

        return y_hat

    def get_params_internal(self, **args):
        """Get the params, which are the trainable variables."""
        del args
        return self._variable_scope.trainable_variables()

    def __getstate__(self):
        """See `Object.__getstate__`."""
        new_dict = super().__getstate__()
        del new_dict['_f_predict']
        return new_dict

    def __setstate__(self, state):
        """See `Object.__setstate__`."""
        super().__setstate__(state)
        self._initialize()
