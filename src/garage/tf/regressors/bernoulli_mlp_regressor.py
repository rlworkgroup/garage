"""Bernoulli MLP Regressor based on MLP with Normalized Inputs."""
from dowel import tabular
import numpy as np
import tensorflow as tf

from garage import make_optimizer
from garage.experiment import deterministic
from garage.tf.distributions import Bernoulli
from garage.tf.misc import tensor_utils
from garage.tf.models import NormalizedInputMLPModel
from garage.tf.optimizers import ConjugateGradientOptimizer, LbfgsOptimizer
from garage.tf.regressors.regressor import StochasticRegressor


class BernoulliMLPRegressor(StochasticRegressor):
    """Fits data to a Bernoulli distribution, parameterized by an MLP.

    Args:
        input_shape (tuple[int]): Input shape of the training data. Since an
            MLP model is used, implementation assumes flattened inputs. The
            input shape of each data point should thus be of shape (x, ).
        output_dim (int): Output dimension of the model.
        name (str): Model name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for the network. For example, (32, 32) means the MLP
            consists of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (Callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (Callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor. Default is Glorot uniform initializer.
        hidden_b_init (Callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor. Default is zero initializer.
        output_nonlinearity (Callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (Callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor. Default is Glorot uniform initializer.
        output_b_init (Callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor. Default is zero initializer.
        optimizer (garage.tf.Optimizer): Optimizer for minimizing the negative
            log-likelihood. Defaults to LbsgsOptimizer
        optimizer_args (dict): Arguments for the optimizer. Default is None,
            which means no arguments.
        tr_optimizer (garage.tf.Optimizer): Optimizer for trust region
            approximation. Defaults to ConjugateGradientOptimizer.
        tr_optimizer_args (dict): Arguments for the trust region optimizer.
            Default is None, which means no arguments.
        use_trust_region (bool): Whether to use trust region constraint.
        max_kl_step (float): KL divergence constraint for each iteration.
        normalize_inputs (bool): Bool for normalizing inputs or not.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 input_shape,
                 output_dim,
                 name='BernoulliMLPRegressor',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=tf.nn.sigmoid,
                 output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 output_b_init=tf.zeros_initializer(),
                 optimizer=None,
                 optimizer_args=None,
                 tr_optimizer=None,
                 tr_optimizer_args=None,
                 use_trust_region=True,
                 max_kl_step=0.01,
                 normalize_inputs=True,
                 layer_normalization=False):

        super().__init__(input_shape, output_dim, name)
        self._use_trust_region = use_trust_region
        self._max_kl_step = max_kl_step
        self._normalize_inputs = normalize_inputs

        with tf.compat.v1.variable_scope(self._name, reuse=False) as vs:
            self._variable_scope = vs
            optimizer_args = optimizer_args or dict()
            tr_optimizer_args = tr_optimizer_args or dict()

            if optimizer is None:
                self._optimizer = make_optimizer(LbfgsOptimizer,
                                                 **optimizer_args)
            else:
                self._optimizer = make_optimizer(optimizer, **optimizer_args)

            if tr_optimizer is None:
                self._tr_optimizer = make_optimizer(ConjugateGradientOptimizer,
                                                    **tr_optimizer_args)
            else:
                self._tr_optimizer = make_optimizer(tr_optimizer,
                                                    **tr_optimizer_args)
            self._first_optimized = False

        self.model = NormalizedInputMLPModel(
            input_shape,
            output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

        self._dist = Bernoulli(output_dim)
        self._network = None

        self._initialize()

    def _initialize(self):
        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, ) +
                                             self._input_shape)

        with tf.compat.v1.variable_scope(self._variable_scope):
            self._network = self.model.build(input_var)

            ys_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                              name='ys',
                                              shape=(None, self._output_dim))

            old_prob_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                                    name='old_prob',
                                                    shape=(None,
                                                           self._output_dim))

            y_hat = self._network.y_hat

            old_info_vars = dict(p=old_prob_var)
            info_vars = dict(p=y_hat)

            mean_kl = tf.reduce_mean(
                self._dist.kl_sym(old_info_vars, info_vars))

            loss = -tf.reduce_mean(
                self._dist.log_likelihood_sym(ys_var, info_vars))

            predicted = y_hat >= 0.5

            self._f_predict = tensor_utils.compile_function([input_var],
                                                            predicted)
            self._f_prob = tensor_utils.compile_function([input_var], y_hat)

            self._optimizer.update_opt(loss=loss,
                                       target=self,
                                       inputs=[input_var, ys_var])
            self._tr_optimizer.update_opt(
                loss=loss,
                target=self,
                inputs=[input_var, ys_var, old_prob_var],
                leq_constraint=(mean_kl, self._max_kl_step))

    def fit(self, xs, ys):
        """Fit with input data xs and label ys.

        Args:
            xs (numpy.ndarray): Input data.
            ys (numpy.ndarray): Label of input data.

        """
        if self._normalize_inputs:
            # recompute normalizing constants for inputs
            self._network.x_mean.load(np.mean(xs, axis=0, keepdims=True))
            self._network.x_std.load(np.std(xs, axis=0, keepdims=True) + 1e-8)

        if self._use_trust_region and self._first_optimized:
            # To use trust region constraint and optimizer
            old_prob = self._f_prob(xs)
            inputs = [xs, ys, old_prob]
            optimizer = self._tr_optimizer
        else:
            inputs = [xs, ys]
            optimizer = self._optimizer
        loss_before = optimizer.loss(inputs)
        tabular.record('{}/LossBefore'.format(self._name), loss_before)
        optimizer.optimize(inputs)
        loss_after = optimizer.loss(inputs)
        tabular.record('{}/LossAfter'.format(self._name), loss_after)
        tabular.record('{}/dLoss'.format(self._name), loss_before - loss_after)
        self._first_optimized = True

    def predict(self, xs):
        """Predict ys based on input xs.

        Args:
            xs (numpy.ndarray): Input data of shape (samples, input_dim)

        Return:
            numpy.ndarray: The deterministic predicted ys (one hot vectors)
                of shape (samples, output_dim)

        """
        return self._f_predict(xs)

    def sample_predict(self, xs):
        """Do a Bernoulli sampling given input xs.

        Args:
            xs (numpy.ndarray): Input data of shape (samples, input_dim)

        Returns:
            numpy.ndarray: The stochastic sampled ys
                of shape (samples, output_dim)

        """
        p = self._f_prob(xs)
        return self._dist.sample(dict(p=p))

    def predict_log_likelihood(self, xs, ys):
        """Log likelihood of ys given input xs.

        Args:
            xs (numpy.ndarray): Input data of shape (samples, input_dim)
            ys (numpy.ndarray): Output data of shape (samples, output_dim)

        Returns:
            numpy.ndarray: The log likelihood of shape (samples, )

        """
        p = self._f_prob(xs)
        return self._dist.log_likelihood(ys, dict(p=p))

    def log_likelihood_sym(self, x_var, y_var, name=None):
        """Build a symbolic graph of the log-likelihood.

        Args:
            x_var (tf.Tensor): Input tf.Tensor for the input data.
            y_var (tf.Tensor): Input tf.Tensor for the one hot label of data.
            name (str): Name of the new graph.

        Return:
            tf.Tensor: Output of the symbolic log-likelihood graph.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            prob, _, _ = self.model.build(x_var, name=name).outputs

        return self._dist.log_likelihood_sym(y_var, dict(p=prob))

    # pylint: disable=unused-argument
    def dist_info_sym(self, input_var, state_info_vars=None, name=None):
        """Build a symbolic graph of the distribution parameters.

        Args:
            input_var (tf.Tensor): Input tf.Tensor for the input data.
            state_info_vars (dict): a dictionary whose values should contain
                information about the state of the regressor at the time it
                received the input.
            name (str): Name of the new graph.

        Return:
            dict[tf.Tensor]: Output of the symbolic graph of the distribution
                parameters.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            prob, _, _ = self.model.build(input_var, name=name).outputs

        return dict(prob=prob)

    @property
    def recurrent(self):
        """bool: If this module has a hidden state."""
        return False

    @property
    def vectorized(self):
        """bool: If this module supports vectorization input."""
        return True

    @property
    def distribution(self):
        """garage.tf.distributions.DiagonalGaussian: Distribution."""
        return self._dist

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_predict']
        del new_dict['_f_prob']
        del new_dict['_network']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
