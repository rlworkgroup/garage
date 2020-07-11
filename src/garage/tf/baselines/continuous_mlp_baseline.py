"""A value function (baseline) based on a MLP model."""
from dowel import tabular
import numpy as np
import tensorflow as tf

from garage import make_optimizer
from garage.experiment import deterministic
from garage.np.baselines import Baseline
from garage.tf.misc import tensor_utils
from garage.tf.models import NormalizedInputMLPModel
from garage.tf.optimizers import LbfgsOptimizer


# pylint: disable=too-many-ancestors
class ContinuousMLPBaseline(NormalizedInputMLPModel, Baseline):
    """A value function using a MLP network.

    It fits the input data by performing linear regression
    to the outputs.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        num_seq_inputs (float): Number of sequence per input. By default
            it is 1.0, which means only one single sequence.
        name (str): Name of baseline.
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
                 env_spec,
                 num_seq_inputs=1,
                 name='ContinuousMLPBaseline',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 output_b_init=tf.zeros_initializer(),
                 optimizer=None,
                 optimizer_args=None,
                 normalize_inputs=True):
        self._env_spec = env_spec
        self._normalize_inputs = normalize_inputs
        self._name = name

        if optimizer_args is None:
            optimizer_args = dict()
        if optimizer is None:
            self._optimizer = make_optimizer(LbfgsOptimizer, **optimizer_args)
        else:
            self._optimizer = make_optimizer(optimizer, **optimizer_args)

        super().__init__(input_shape=(env_spec.observation_space.flat_dim *
                                      num_seq_inputs, ),
                         output_dim=1,
                         name=name,
                         hidden_sizes=hidden_sizes,
                         hidden_nonlinearity=hidden_nonlinearity,
                         hidden_w_init=hidden_w_init,
                         hidden_b_init=hidden_b_init,
                         output_nonlinearity=output_nonlinearity,
                         output_w_init=output_w_init,
                         output_b_init=output_b_init)

        self._x_mean = None
        self._x_std = None
        self._y_hat = None
        self._initialize()

    def _initialize(self):
        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, ) +
                                             self._input_shape)

        ys_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                          name='ys',
                                          shape=(None, self._output_dim))

        (self._y_hat, self._x_mean,
         self._x_std) = self.build(input_var).outputs

        loss = tf.reduce_mean(tf.square(self._y_hat - ys_var))
        self._f_predict = tensor_utils.compile_function([input_var],
                                                        self._y_hat)
        optimizer_args = dict(
            loss=loss,
            target=self,
            network_outputs=[ys_var],
        )
        optimizer_args['inputs'] = [input_var, ys_var]
        with tf.name_scope('update_opt'):
            self._optimizer.update_opt(**optimizer_args)

    def fit(self, paths):
        """Fit regressor based on paths.

        Args:
            paths (dict[numpy.ndarray]): Sample paths.

        """
        xs = np.concatenate([p['observations'] for p in paths])
        ys = np.concatenate([p['returns'] for p in paths])
        ys = ys.reshape((-1, 1))
        if self._normalize_inputs:
            # recompute normalizing constants for inputs
            self._x_mean.load(np.mean(xs, axis=0, keepdims=True))
            self._x_std.load(np.std(xs, axis=0, keepdims=True) + 1e-8)

        inputs = [xs, ys]
        loss_before = self._optimizer.loss(inputs)
        tabular.record('{}/LossBefore'.format(self._name), loss_before)
        self._optimizer.optimize(inputs)
        loss_after = self._optimizer.loss(inputs)
        tabular.record('{}/LossAfter'.format(self._name), loss_after)
        tabular.record('{}/dLoss'.format(self._name), loss_before - loss_after)

    def predict(self, paths):
        """Predict value based on paths.

        Args:
            paths (dict[numpy.ndarray]): Sample paths.

        Returns:
            numpy.ndarray: Predicted value.

        """
        return self._f_predict(paths['observations']).flatten()

    @property
    def recurrent(self):
        """bool: If this module has a hidden state."""
        return False

    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """
        return self._env_spec

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_predict']
        del new_dict['_x_mean']
        del new_dict['_x_std']
        del new_dict['_y_hat']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
