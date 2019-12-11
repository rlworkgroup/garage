"""Penalized Limited-memory BFGS (L-BFGS) optimizer."""
from dowel import logger
import numpy as np
import scipy.optimize
import tensorflow as tf

from garage.tf.misc import tensor_utils
from garage.tf.optimizers.utils import LazyDict


class PenaltyLbfgsOptimizer:
    """Penalized Limited-memory BFGS (L-BFGS) optimizer.

    Performs constrained optimization via penalized L-BFGS. The penalty term is
    adaptively adjusted to make sure that the constraint is satisfied.

    Args:
        max_opt_itr (int): Maximum iteration for update.
        initial_penalty (float): Initial penalty.
        min_penalty (float): Minimum penalty allowed. Penalty will be clipped
            if lower than this value.
        max_penalty (float): Maximum penalty allowed. Penalty will be clipped
            if higher than this value.
        increase_penalty_factor (float): Factor to increase penalty in each
            penalty iteration.
        decrease_penalty_factor (float): Factor to decrease penalty in each
            penalty iteration.
        max_penalty_itr (int): Maximum penalty iterations to perform.
        adapt_penalty (bool): Whether the penalty is adaptive or not. If false,
            penalty will not change.

    """

    def __init__(self,
                 max_opt_itr=20,
                 initial_penalty=1.0,
                 min_penalty=1e-2,
                 max_penalty=1e6,
                 increase_penalty_factor=2,
                 decrease_penalty_factor=0.5,
                 max_penalty_itr=10,
                 adapt_penalty=True):
        self._max_opt_itr = max_opt_itr
        self._penalty = initial_penalty
        self._initial_penalty = initial_penalty
        self._min_penalty = min_penalty
        self._max_penalty = max_penalty
        self._increase_penalty_factor = increase_penalty_factor
        self._decrease_penalty_factor = decrease_penalty_factor
        self._max_penalty_itr = max_penalty_itr
        self._adapt_penalty = adapt_penalty

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None

    def update_opt(self,
                   loss,
                   target,
                   leq_constraint,
                   inputs,
                   constraint_name='constraint',
                   name='PenaltyLbfgsOptimizer',
                   **kwargs):
        """Construct operation graph for the optimizer.

        Args:
            loss (tf.Tensor): Loss objective to minimize.
            target (object): Target object to optimize. The object should
                implemenet `get_params()` and `get_param_values`.
            leq_constraint (tuple): It contains a tf.Tensor and a float value.
                The tf.Tensor represents the constraint term, and the float
                value is the constraint value.
            inputs (list[tf.Tensor]): List of input placeholders.
            constraint_name (str): Constraint name for logging.
            name (str): Name scope.
            kwargs (dict): Extra unused keyword arguments. Some optimizers
                have extra input, e.g. KL constraint.

        """
        del kwargs
        params = target.get_params()
        with tf.name_scope(name):
            constraint_term, constraint_value = leq_constraint
            penalty_var = tf.compat.v1.placeholder(tf.float32,
                                                   tuple(),
                                                   name='penalty')
            penalized_loss = loss + penalty_var * constraint_term

            self._target = target
            self._max_constraint_val = constraint_value
            self._constraint_name = constraint_name

            def get_opt_output():
                """Helper function to construct graph.

                Returns:
                    list[tf.Tensor]: Penalized loss and gradient tensor.

                """
                with tf.name_scope('get_opt_output'):
                    grads = tf.gradients(penalized_loss, params)
                    for idx, (grad, param) in enumerate(zip(grads, params)):
                        if grad is None:
                            grads[idx] = tf.zeros_like(param)
                    flat_grad = tensor_utils.flatten_tensor_variables(grads)
                    return [
                        tf.cast(penalized_loss, tf.float64),
                        tf.cast(flat_grad, tf.float64),
                    ]

            self._opt_fun = LazyDict(
                f_loss=lambda: tensor_utils.compile_function(
                    inputs, loss, log_name='f_loss'),
                f_constraint=lambda: tensor_utils.compile_function(
                    inputs, constraint_term, log_name='f_constraint'),
                f_penalized_loss=lambda: tensor_utils.compile_function(
                    inputs=inputs + [penalty_var],
                    outputs=[penalized_loss, loss, constraint_term],
                    log_name='f_penalized_loss',
                ),
                f_opt=lambda: tensor_utils.compile_function(
                    inputs=inputs + [penalty_var],
                    outputs=get_opt_output(),
                ))

    def loss(self, inputs):
        """The loss.

        Args:
            inputs (list[numpy.ndarray]): List of input values.

        Returns:
            float: Loss.

        Raises:
            Exception: If loss function is None, i.e. not defined.

        """
        if self._opt_fun is None:
            raise Exception(
                'Use update_opt() to setup the loss function first.')
        return self._opt_fun['f_loss'](*inputs)

    def constraint_val(self, inputs):
        """The constraint value.

        Args:
            inputs (list[numpy.ndarray]): List of input values.

        Returns:
            float: Constraint value.

        Raises:
            Exception: If loss function is None, i.e. not defined.

        """
        if self._opt_fun is None:
            raise Exception(
                'Use update_opt() to setup the loss function first.')
        return self._opt_fun['f_constraint'](*inputs)

    def optimize(self, inputs, name='optimize'):
        """Perform optimization.

        Args:
            inputs (list[numpy.ndarray]): List of input values.
            name (str): Name scope.

        Raises:
            Exception: If loss function is None, i.e. not defined.

        """
        if self._opt_fun is None:
            raise Exception(
                'Use update_opt() to setup the loss function first.')

        with tf.name_scope(name):

            inputs = tuple(inputs)

            try_penalty = np.clip(self._penalty, self._min_penalty,
                                  self._max_penalty)

            penalty_scale_factor = None
            f_opt = self._opt_fun['f_opt']
            f_penalized_loss = self._opt_fun['f_penalized_loss']

            def gen_f_opt(penalty):  # noqa: D202
                """Return a function that set parameters values.

                Args:
                    penalty (float): Penalty.

                Returns:
                    callable: Function that set parameters values.

                """

                def f(flat_params):
                    """Helper function to set parameters values.

                    Args:
                        flat_params (numpy.ndarray): Flatten parameter values.

                    Returns:
                        list[tf.Tensor]: Penalized loss and gradient tensor.

                    """
                    self._target.set_param_values(flat_params)
                    return f_opt(*(inputs + (penalty, )))

                return f

            cur_params = self._target.get_param_values().astype('float64')
            opt_params = cur_params

            for penalty_itr in range(self._max_penalty_itr):
                logger.log('trying penalty=%.3f...' % try_penalty)

                itr_opt_params, _, _ = scipy.optimize.fmin_l_bfgs_b(
                    func=gen_f_opt(try_penalty),
                    x0=cur_params,
                    maxiter=self._max_opt_itr)

                _, try_loss, try_constraint_val = f_penalized_loss(*(
                    inputs + (try_penalty, )))

                logger.log('penalty %f => loss %f, %s %f' %
                           (try_penalty, try_loss, self._constraint_name,
                            try_constraint_val))

                # Either constraint satisfied, or we are at the last iteration
                # already and no alternative parameter satisfies the constraint
                if try_constraint_val < self._max_constraint_val or \
                        (penalty_itr == self._max_penalty_itr - 1 and
                            opt_params is None):
                    opt_params = itr_opt_params

                if not self._adapt_penalty:
                    break

                # Decide scale factor on the first iteration, or if constraint
                # violation yields numerical error
                if (penalty_scale_factor is None
                        or np.isnan(try_constraint_val)):
                    # Increase penalty if constraint violated, or if constraint
                    # term is NAN
                    if (try_constraint_val > self._max_constraint_val
                            or np.isnan(try_constraint_val)):
                        penalty_scale_factor = self._increase_penalty_factor
                    else:
                        # Otherwise (i.e. constraint satisfied), shrink penalty
                        penalty_scale_factor = self._decrease_penalty_factor
                        opt_params = itr_opt_params
                else:
                    if (penalty_scale_factor > 1 and
                            try_constraint_val <= self._max_constraint_val):
                        break
                    if (penalty_scale_factor < 1 and
                            try_constraint_val >= self._max_constraint_val):
                        break
                try_penalty *= penalty_scale_factor
                try_penalty = np.clip(try_penalty, self._min_penalty,
                                      self._max_penalty)
                self._penalty = try_penalty

            self._target.set_param_values(opt_params)

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = self.__dict__.copy()
        del new_dict['_opt_fun']
        return new_dict
