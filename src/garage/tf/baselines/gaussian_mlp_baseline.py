"""A value function (baseline) based on a GaussianMLP model."""
from dowel import tabular
import numpy as np
import tensorflow as tf

from garage import make_optimizer
from garage.experiment import deterministic
from garage.np.baselines import Baseline
from garage.tf import compile_function
from garage.tf.baselines.gaussian_mlp_baseline_model import (
    GaussianMLPBaselineModel)
from garage.tf.optimizers import LbfgsOptimizer, PenaltyLbfgsOptimizer


# pylint: disable=too-many-ancestors
class GaussianMLPBaseline(GaussianMLPBaselineModel, Baseline):
    """Gaussian MLP Baseline with Model.

    It fits the input data to a gaussian distribution estimated by
    a MLP.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        subsample_factor (float): The factor to subsample the data. By
            default it is 1.0, which means using all the data.
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
        use_trust_region (bool): Whether to use trust region constraint.
        max_kl_step (float): KL divergence constraint for each iteration.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
        adaptive_std (bool): Is std a neural network. If False, it will be a
            parameter.
        std_share_network (bool): Boolean for whether mean and std share
            the same network.
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        std_nonlinearity (Callable): Nonlinearity for each hidden layer in
            the std network.
        layer_normalization (bool): Bool for using layer normalization or not.
        normalize_inputs (bool): Bool for normalizing inputs or not.
        normalize_outputs (bool): Bool for normalizing outputs or not.
        subsample_factor (float): The factor to subsample the data. By default
            it is 1.0, which means using all the data.

    """

    def __init__(self,
                 env_spec,
                 num_seq_inputs=1,
                 name='GaussianMLPBaseline',
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
                 use_trust_region=True,
                 max_kl_step=0.01,
                 learn_std=True,
                 init_std=1.0,
                 adaptive_std=False,
                 std_share_network=False,
                 std_hidden_sizes=(32, 32),
                 std_nonlinearity=None,
                 layer_normalization=False,
                 normalize_inputs=True,
                 normalize_outputs=True,
                 subsample_factor=1.0):
        self._env_spec = env_spec
        self._num_seq_inputs = num_seq_inputs
        self._use_trust_region = use_trust_region
        self._max_kl_step = max_kl_step
        self._normalize_inputs = normalize_inputs
        self._normalize_outputs = normalize_outputs
        self._subsample_factor = subsample_factor

        if optimizer_args is None:
            optimizer_args = dict()
        if optimizer is None:
            if use_trust_region:
                self._optimizer = make_optimizer(PenaltyLbfgsOptimizer,
                                                 **optimizer_args)
            else:
                self._optimizer = make_optimizer(LbfgsOptimizer,
                                                 **optimizer_args)
        else:
            self._optimizer = make_optimizer(optimizer, **optimizer_args)

        super().__init__(name=name,
                         input_shape=(env_spec.observation_space.flat_dim *
                                      num_seq_inputs, ),
                         output_dim=1,
                         hidden_sizes=hidden_sizes,
                         hidden_nonlinearity=hidden_nonlinearity,
                         hidden_w_init=hidden_w_init,
                         hidden_b_init=hidden_b_init,
                         output_nonlinearity=output_nonlinearity,
                         output_w_init=output_w_init,
                         output_b_init=output_b_init,
                         learn_std=learn_std,
                         adaptive_std=adaptive_std,
                         std_share_network=std_share_network,
                         init_std=init_std,
                         min_std=None,
                         max_std=None,
                         std_hidden_sizes=std_hidden_sizes,
                         std_hidden_nonlinearity=std_nonlinearity,
                         std_output_nonlinearity=None,
                         std_parameterization='exp',
                         layer_normalization=layer_normalization)
        # model for old distribution, used when trusted region is on
        self._old_model = self.clone_model(name=name + '_old_model')
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None
        self._old_network = None

        self._initialize()

    def _initialize(self):
        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, ) +
                                             self._input_shape)
        ys_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                          name='ys',
                                          shape=(None, self._output_dim))

        self._old_network = self._old_model.build(input_var)
        (norm_dist, norm_mean, norm_log_std, _, mean, _, self._x_mean,
         self._x_std, self._y_mean,
         self._y_std) = self.build(input_var).outputs

        normalized_ys_var = (ys_var - self._y_mean) / self._y_std
        old_normalized_dist = self._old_network.normalized_dist

        mean_kl = tf.reduce_mean(old_normalized_dist.kl_divergence(norm_dist))
        loss = -tf.reduce_mean(norm_dist.log_prob(normalized_ys_var))
        self._f_predict = compile_function([input_var], mean)
        optimizer_args = dict(
            loss=loss,
            target=self,
            network_outputs=[norm_mean, norm_log_std],
        )
        if self._use_trust_region:
            optimizer_args['leq_constraint'] = (mean_kl, self._max_kl_step)
        optimizer_args['inputs'] = [input_var, ys_var]
        with tf.name_scope('update_opt'):
            self._optimizer.update_opt(**optimizer_args)

    # pylint: disable=unsubscriptable-object
    def fit(self, paths):
        """Fit regressor based on paths.

        Args:
            paths (list[dict]): Sample paths.

        """
        xs = np.concatenate([p['observations'] for p in paths])
        ys = np.concatenate([p['returns'] for p in paths])
        ys = ys.reshape((-1, 1))

        if self._subsample_factor < 1:
            num_samples_tot = xs.shape[0]
            idx = np.random.randint(
                0, num_samples_tot,
                int(num_samples_tot * self._subsample_factor))
            xs, ys = xs[idx], ys[idx]

        if self._normalize_inputs:
            # recompute normalizing constants for inputs
            self._x_mean.load(np.mean(xs, axis=0, keepdims=True))
            self._x_std.load(np.std(xs, axis=0, keepdims=True) + 1e-8)
            self._old_network.x_mean.load(np.mean(xs, axis=0, keepdims=True))
            self._old_network.x_std.load(
                np.std(xs, axis=0, keepdims=True) + 1e-8)
        if self._normalize_outputs:
            # recompute normalizing constants for outputs
            self._y_mean.load(np.mean(ys, axis=0, keepdims=True))
            self._y_std.load(np.std(ys, axis=0, keepdims=True) + 1e-8)
            self._old_network.y_mean.load(np.mean(ys, axis=0, keepdims=True))
            self._old_network.y_std.load(
                np.std(ys, axis=0, keepdims=True) + 1e-8)
        inputs = [xs, ys]
        loss_before = self._optimizer.loss(inputs)
        tabular.record('{}/LossBefore'.format(self._name), loss_before)
        self._optimizer.optimize(inputs)
        loss_after = self._optimizer.loss(inputs)
        tabular.record('{}/LossAfter'.format(self._name), loss_after)
        if self._use_trust_region:
            tabular.record('{}/MeanKL'.format(self._name),
                           self._optimizer.constraint_val(inputs))
        tabular.record('{}/dLoss'.format(self._name), loss_before - loss_after)
        self._old_model.parameters = self.parameters

    def predict(self, paths):
        """Predict value based on paths.

        Args:
            paths (list[dict]): Sample paths.

        Returns:
            numpy.ndarray: Predicted value.

        """
        return self._f_predict(paths['observations']).flatten()

    def clone_model(self, name):
        """Return a clone of the GaussianMLPBaselineModel.

        It copies the configuration of the primitive and also the parameters.

        Args:
            name (str): Name of the newly created model. It has to be
                different from source policy if cloned under the same
                computational graph.

        Returns:
            garage.tf.baselines.GaussianMLPBaselineModel: Newly cloned model.

        """
        new_baseline = GaussianMLPBaselineModel(
            name=name,
            input_shape=(self._env_spec.observation_space.flat_dim *
                         self._num_seq_inputs, ),
            output_dim=1,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            learn_std=self._learn_std,
            adaptive_std=self._adaptive_std,
            std_share_network=self._std_share_network,
            init_std=self._init_std,
            min_std=None,
            max_std=None,
            std_hidden_sizes=self._std_hidden_sizes,
            std_hidden_nonlinearity=self._std_hidden_nonlinearity,
            std_output_nonlinearity=None,
            std_parameterization='exp',
            layer_normalization=self._layer_normalization)
        new_baseline.parameters = self.parameters
        return new_baseline

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
            dict: The state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_predict']
        del new_dict['_x_mean']
        del new_dict['_x_std']
        del new_dict['_y_mean']
        del new_dict['_y_std']
        del new_dict['_old_network']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
