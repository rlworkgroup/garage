"""A regressor based on MLP with Normalized Inputs."""
from dowel import tabular
import numpy as np
import tensorflow as tf

from garage.tf.distributions import Categorical
from garage.tf.misc import tensor_utils
from garage.tf.models import NormalizedInputMLPModel
from garage.tf.optimizers import ConjugateGradientOptimizer, LbfgsOptimizer
from garage.tf.regressors import StochasticRegressor2


class CategoricalMLPRegressorWithModel(StochasticRegressor2):
    """
    CategoricalMLPRegressor with garage.tf.models.NormalizedInputMLPModel.

    A class for performing regression (or classification, really) by fitting
    a Categorical distribution to the outputs. Assumes that the output will
    always be a one hot vector

    Args:
        input_shape (tuple[int]): Input shape of the training data.
        output_dim (int): Output dimension of the model.
        name (str): Model name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for the network. For example, (32, 32) means the MLP
            consists of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a tanh activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor. Default is Glorot uniform initializer.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor. Default is zero initializer.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a softmax activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor. Default is Glorot uniform initializer.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor. Default is zero initializer.
        optimizer (tf.Optimizer): Optimizer for minimizing the negative
            log-likelihood.
        optimizer_args (dict): Arguments for the optimizer. Default is None,
            which means no arguments.
        tr_optimizer (tf.Optimizer): Optimizer for trust region approximation.
        tr_optimizer_args (dict): Arguments for the trust region optimizer.
            Default is None, which means no arguments.
        use_trust_region (bool): Whether to use trust region constraint.
        max_kl_step (float): KL divergence constraint for each iteration.
        normalize_inputs (bool): Bool for normalizing inputs or not.
        layer_normalization (bool): Bool for using layer normalization or not.
        no_initial_trust_region (bool): Bool to specify if trust region
            constraint has been used initially.
    """

    def __init__(self,
                 input_shape,
                 output_dim,
                 name='CategoricalMLPRegressorWithModel',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=tf.nn.softmax,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 optimizer=None,
                 optimizer_args=None,
                 tr_optimizer=None,
                 tr_optimizer_args=None,
                 use_trust_region=True,
                 max_kl_step=0.01,
                 normalize_inputs=True,
                 layer_normalization=False,
                 no_initial_trust_region=True):

        super().__init__(input_shape, output_dim, name)
        self._use_trust_region = use_trust_region
        self._no_initial_trust_region = no_initial_trust_region
        self._max_kl_step = max_kl_step
        self._normalize_inputs = normalize_inputs
        self._first_optimized = not no_initial_trust_region

        with tf.variable_scope(
                self._name, reuse=False) as self._variable_scope:
            if optimizer is None:
                optimizer = LbfgsOptimizer()
            if tr_optimizer is None:
                tr_optimizer = ConjugateGradientOptimizer()
            self._optimizer = optimizer
            self._tr_optimizer = tr_optimizer

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

        self._initialize()

    def _initialize(self):
        input_var = tf.placeholder(
            tf.float32, shape=(None, ) + self._input_shape)

        with tf.variable_scope(self._variable_scope):
            self.model.build(input_var)

            ys_var = tf.placeholder(
                dtype=tf.float32, name='ys', shape=(None, self._output_dim))

            old_prob_var = tf.placeholder(
                dtype=tf.float32,
                name='old_prob',
                shape=(None, self._output_dim))

            y_hat = self.model.networks['default'].y_hat

            old_info_vars = dict(prob=old_prob_var)
            info_vars = dict(prob=y_hat)

            self._dist = Categorical(self._output_dim)
            mean_kl = tf.reduce_mean(
                self._dist.kl_sym(old_info_vars, info_vars))

            loss = -tf.reduce_mean(
                self._dist.log_likelihood_sym(ys_var, info_vars))

            predicted = tensor_utils.to_onehot_sym(
                tf.argmax(y_hat, axis=1), self._output_dim)

            self._f_predict = tensor_utils.compile_function([input_var],
                                                            predicted)
            self._f_prob = tensor_utils.compile_function([input_var], y_hat)

            self._optimizer.update_opt(
                loss=loss,
                target=self,
                network_output=[y_hat],
                inputs=[input_var, ys_var])
            self._tr_optimizer.update_opt(
                loss=loss,
                target=self,
                network_output=[y_hat],
                inputs=[input_var, ys_var, old_prob_var],
                leq_constraint=(mean_kl, self._max_kl_step))

    def fit(self, xs, ys):
        """
        Fit with input data xs and label ys.

        Args:
            xs (numpy.ndarray): Input data.
            ys (numpy.ndarray): Label of input data.
        """
        if self._normalize_inputs:
            # recompute normalizing constants for inputs
            self.model.networks['default'].x_mean.load(
                np.mean(xs, axis=0, keepdims=True))
            self.model.networks['default'].x_std.load(
                np.std(xs, axis=0, keepdims=True))

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
        self.first_optimized = True

    def predict(self, xs):
        """
        Predict ys based on input xs.

        Args:
            xs (numpy.ndarray): Input data.

        Return:
            The predicted ys (one hot vectors).
        """
        return self._f_predict(xs)

    def predict_log_likelihood(self, xs, ys):
        """
        Predict log likelihood of output based on input xs and labels ys.

        Args:
            xs (numpy.ndarray): Input data.
            ys (numpy.ndarray): Input labels in one hot representation.

        Return:
            The predicted log likelihoods.

        """
        prob = self._f_prob(np.asarray(xs))
        return self._dist.log_likelihood(np.asarray(ys), dict(prob=prob))

    def dist_info_sym(self, x_var, name=None):
        """Symbolic graph of the distribution."""
        with tf.variable_scope(self._variable_scope):
            prob, _, _ = self.model.build(x_var, name=name)

        return dict(prob=prob)

    def log_likelihood_sym(self, x_var, y_var, name=None):
        """
        Symbolic graph of the log likelihood.

        Args:
            x_var (tf.Tensor): Input tf.Tensor for the input data.
            y_var (tf.Tensor): Input tf.Tensor for the one hot label of data.
            name (str): Name of the new graph.

        Return:
            tf.Tensor output of the symbolic log likelihood.
        """
        with tf.variable_scope(self._variable_scope):
            prob, _, _ = self.model.build(x_var, name=name)

        return self._dist.log_likelihood_sym(y_var, dict(prob=prob))

    def get_params_internal(self, **args):
        """Get the params, which are the trainable variables."""
        return self._variable_scope.trainable_variables()

    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = super().__getstate__()
        del new_dict['_f_predict']
        del new_dict['_f_prob']
        del new_dict['_dist']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        super().__setstate__(state)
        self._initialize()
