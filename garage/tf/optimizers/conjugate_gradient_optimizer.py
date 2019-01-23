import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.logger import logger
from garage.misc import krylov
from garage.tf.misc import tensor_utils
from garage.tf.optimizers.utils import LazyDict, sliced_fun


class PerlmutterHvp:
    def __init__(self, num_slices=1):
        self.target = None
        self.reg_coeff = None
        self.opt_fun = None
        self._num_slices = num_slices

    def update_opt(self, f, target, inputs, reg_coeff, name=None):
        self.target = target
        self.reg_coeff = reg_coeff
        params = target.get_params(trainable=True)
        with tf.name_scope(name, "PerlmutterHvp", [f, inputs, params]):
            constraint_grads = tf.gradients(
                f, xs=params, name="gradients_constraint")
            for idx, (grad, param) in enumerate(zip(constraint_grads, params)):
                if grad is None:
                    constraint_grads[idx] = tf.zeros_like(param)

            xs = tuple([
                tensor_utils.new_tensor_like(p.name.split(":")[0], p)
                for p in params
            ])

            def hx_plain():
                with tf.name_scope(
                        "hx_plain", values=[constraint_grads, params, xs]):
                    with tf.name_scope(
                            "hx_function", values=[constraint_grads, xs]):
                        hx_f = tf.reduce_sum(
                            tf.stack([
                                tf.reduce_sum(g * x)
                                for g, x in zip(constraint_grads, xs)
                            ])),
                    hx_plain_splits = tf.gradients(
                        hx_f, params, name="gradients_hx_plain")
                    for idx, (hx, param) in enumerate(
                            zip(hx_plain_splits, params)):
                        if hx is None:
                            hx_plain_splits[idx] = tf.zeros_like(param)
                    return tensor_utils.flatten_tensor_variables(
                        hx_plain_splits)

            self.opt_fun = LazyDict(
                f_hx_plain=lambda: tensor_utils.compile_function(
                    inputs=inputs + xs,
                    outputs=hx_plain(),
                    log_name="f_hx_plain",
                ), )

    def build_eval(self, inputs):
        def eval(x):
            xs = tuple(self.target.flat_to_params(x, trainable=True))
            ret = sliced_fun(self.opt_fun["f_hx_plain"], self._num_slices)(
                inputs, xs) + self.reg_coeff * x
            return ret

        return eval


class FiniteDifferenceHvp:
    def __init__(self,
                 base_eps=1e-8,
                 symmetric=True,
                 grad_clip=None,
                 num_slices=1):
        self.base_eps = base_eps
        self.symmetric = symmetric
        self.grad_clip = grad_clip
        self._num_slices = num_slices

    def update_opt(self, f, target, inputs, reg_coeff, name=None):
        self.target = target
        self.reg_coeff = reg_coeff
        params = target.get_params(trainable=True)

        with tf.name_scope(name, "FiniteDifferenceHvp",
                           [f, inputs, params, target]):
            constraint_grads = tf.gradients(
                f, xs=params, name="gradients_constraint")
            for idx, (grad, param) in enumerate(zip(constraint_grads, params)):
                if grad is None:
                    constraint_grads[idx] = tf.zeros_like(param)

            flat_grad = tensor_utils.flatten_tensor_variables(constraint_grads)

            def f_hx_plain(*args):
                with tf.name_scope("f_hx_plain", values=[inputs, self.target]):
                    inputs_ = args[:len(inputs)]
                    xs = args[len(inputs):]
                    flat_xs = np.concatenate(
                        [np.reshape(x, (-1, )) for x in xs])
                    param_val = self.target.get_param_values(trainable=True)
                    eps = np.cast['float32'](
                        self.base_eps / (np.linalg.norm(param_val) + 1e-8))
                    self.target.set_param_values(
                        param_val + eps * flat_xs, trainable=True)
                    flat_grad_dvplus = self.opt_fun["f_grad"](*inputs_)
                    self.target.set_param_values(param_val, trainable=True)
                    if self.symmetric:
                        self.target.set_param_values(
                            param_val - eps * flat_xs, trainable=True)
                        flat_grad_dvminus = self.opt_fun["f_grad"](*inputs_)
                        hx = (flat_grad_dvplus - flat_grad_dvminus) / (2 * eps)
                        self.target.set_param_values(param_val, trainable=True)
                    else:
                        flat_grad = self.opt_fun["f_grad"](*inputs_)
                        hx = (flat_grad_dvplus - flat_grad) / eps
                    return hx

            self.opt_fun = LazyDict(
                f_grad=lambda: tensor_utils.compile_function(
                    inputs=inputs,
                    outputs=flat_grad,
                    log_name="f_grad",
                ),
                f_hx_plain=lambda: f_hx_plain,
            )

    def build_eval(self, inputs):
        def eval(x):
            xs = tuple(self.target.flat_to_params(x, trainable=True))
            ret = sliced_fun(self.opt_fun["f_hx_plain"], self._num_slices)(
                inputs, xs) + self.reg_coeff * x
            return ret

        return eval


class ConjugateGradientOptimizer(Serializable):
    """Performs constrained optimization via line search.

    The search direction is computed using a conjugate gradient algorithm,
    which gives x = A^{-1}g, where A is a second order approximation of the
    constraint and g is the gradient of the loss function.
    """

    def __init__(self,
                 cg_iters=10,
                 reg_coeff=1e-5,
                 subsample_factor=1.,
                 backtrack_ratio=0.8,
                 max_backtracks=15,
                 debug_nan=False,
                 accept_violation=False,
                 hvp_approach=None,
                 num_slices=1,
                 name="ConjugateGradientOptimizer"):
        """
        :param cg_iters: The number of CG iterations used to calculate A^-1 g
        :param reg_coeff: A small value so that A -> A + reg*I
        :param subsample_factor: Subsampling factor to reduce samples when
         using "conjugate gradient. Since the computation time for the descent
         direction dominates, this can greatly reduce the overall computation
         time.
        :param debug_nan: if set to True, NanGuard will be added to the
         compilation, and ipdb will be invoked when nan is detected
        :param accept_violation: whether to accept the descent step if it
         violates the line search condition after exhausting all backtracking
         budgets
        :return:
        """
        Serializable.quick_init(self, locals())
        self._cg_iters = cg_iters
        self._reg_coeff = reg_coeff
        self._subsample_factor = subsample_factor
        self._backtrack_ratio = backtrack_ratio
        self._max_backtracks = max_backtracks
        self._num_slices = num_slices
        self._name = name

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None
        self._debug_nan = debug_nan
        self._accept_violation = accept_violation
        if hvp_approach is None:
            hvp_approach = PerlmutterHvp(num_slices)
        self._hvp_approach = hvp_approach

    def update_opt(self,
                   loss,
                   target,
                   leq_constraint,
                   inputs,
                   extra_inputs=None,
                   name=None,
                   constraint_name="constraint",
                   *args,
                   **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should
         implement methods of the
         the :class:`garage.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon),
         of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs, which could be
         subsampled if needed. It is assumed that the first dimension of these
         inputs should correspond to the number of data points
        :param extra_inputs: A list of symbolic variables as extra inputs which
         should not be subsampled
        :return: No return value.
        """
        params = target.get_params(trainable=True)
        with tf.name_scope(
                name, "ConjugateGradientOptimizer",
                [loss, target, leq_constraint, inputs, extra_inputs,
                 params]):  # yapf: disable
            inputs = tuple(inputs)
            if extra_inputs is None:
                extra_inputs = tuple()
            else:
                extra_inputs = tuple(extra_inputs)

            constraint_term, constraint_value = leq_constraint

            with tf.name_scope("loss_gradients", values=[loss, params]):
                grads = tf.gradients(loss, xs=params)
                for idx, (grad, param) in enumerate(zip(grads, params)):
                    if grad is None:
                        grads[idx] = tf.zeros_like(param)
                flat_grad = tensor_utils.flatten_tensor_variables(grads)

            self._hvp_approach.update_opt(
                f=constraint_term,
                target=target,
                inputs=inputs + extra_inputs,
                reg_coeff=self._reg_coeff,
                name="update_opt_" + constraint_name)

            self._target = target
            self._max_constraint_val = constraint_value
            self._constraint_name = constraint_name

            self._opt_fun = LazyDict(
                f_loss=lambda: tensor_utils.compile_function(
                    inputs=inputs + extra_inputs,
                    outputs=loss,
                    log_name="f_loss",
                ),
                f_grad=lambda: tensor_utils.compile_function(
                    inputs=inputs + extra_inputs,
                    outputs=flat_grad,
                    log_name="f_grad",
                ),
                f_constraint=lambda: tensor_utils.compile_function(
                    inputs=inputs + extra_inputs,
                    outputs=constraint_term,
                    log_name="constraint",
                ),
                f_loss_constraint=lambda: tensor_utils.compile_function(
                    inputs=inputs + extra_inputs,
                    outputs=[loss, constraint_term],
                    log_name="f_loss_constraint",
                ),
            )

    def loss(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return sliced_fun(self._opt_fun["f_loss"],
                          self._num_slices)(inputs, extra_inputs)

    def constraint_val(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return sliced_fun(self._opt_fun["f_constraint"],
                          self._num_slices)(inputs, extra_inputs)

    def optimize(self,
                 inputs,
                 extra_inputs=None,
                 subsample_grouped_inputs=None,
                 name=None):
        with tf.name_scope(
                name,
                "optimize",
                values=[inputs, extra_inputs, subsample_grouped_inputs]):
            prev_param = np.copy(self._target.get_param_values(trainable=True))
            inputs = tuple(inputs)
            if extra_inputs is None:
                extra_inputs = tuple()

            if self._subsample_factor < 1:
                if subsample_grouped_inputs is None:
                    subsample_grouped_inputs = [inputs]
                subsample_inputs = tuple()
                for inputs_grouped in subsample_grouped_inputs:
                    n_samples = len(inputs_grouped[0])
                    inds = np.random.choice(
                        n_samples,
                        int(n_samples * self._subsample_factor),
                        replace=False)
                    subsample_inputs += tuple(
                        [x[inds] for x in inputs_grouped])
            else:
                subsample_inputs = inputs

            logger.log(
                ("Start CG optimization: "
                 "#parameters: %d, #inputs: %d, #subsample_inputs: %d") %
                (len(prev_param), len(inputs[0]), len(subsample_inputs[0])))

            logger.log("computing loss before")
            loss_before = sliced_fun(self._opt_fun["f_loss"],
                                     self._num_slices)(inputs, extra_inputs)
            logger.log("performing update")

            logger.log("computing gradient")
            flat_g = sliced_fun(self._opt_fun["f_grad"],
                                self._num_slices)(inputs, extra_inputs)
            logger.log("gradient computed")

            logger.log("computing descent direction")
            hx = self._hvp_approach.build_eval(subsample_inputs + extra_inputs)

            descent_direction = krylov.cg(hx, flat_g, cg_iters=self._cg_iters)

            initial_step_size = np.sqrt(
                2.0 * self._max_constraint_val *
                (1. / (descent_direction.dot(hx(descent_direction)) + 1e-8)))
            if np.isnan(initial_step_size):
                initial_step_size = 1.
            flat_descent_step = initial_step_size * descent_direction

            logger.log("descent direction computed")

            n_iter = 0
            for n_iter, ratio in enumerate(self._backtrack_ratio**np.arange(
                    self._max_backtracks)):  # yapf: disable
                cur_step = ratio * flat_descent_step
                cur_param = prev_param - cur_step
                self._target.set_param_values(cur_param, trainable=True)
                loss, constraint_val = sliced_fun(
                    self._opt_fun["f_loss_constraint"],
                    self._num_slices)(inputs, extra_inputs)
                if self._debug_nan and np.isnan(constraint_val):
                    break
                if loss < loss_before and \
                   constraint_val <= self._max_constraint_val:
                    break
            if (np.isnan(loss) or np.isnan(constraint_val)
                    or loss >= loss_before or constraint_val >=
                    self._max_constraint_val) and not self._accept_violation:
                logger.log(
                    "Line search condition violated. Rejecting the step!")
                if np.isnan(loss):
                    logger.log("Violated because loss is NaN")
                if np.isnan(constraint_val):
                    logger.log("Violated because constraint %s is NaN" %
                               self._constraint_name)
                if loss >= loss_before:
                    logger.log("Violated because loss not improving")
                if constraint_val >= self._max_constraint_val:
                    logger.log("Violated because constraint %s is violated" %
                               self._constraint_name)
                self._target.set_param_values(prev_param, trainable=True)
            logger.log("backtrack iters: %d" % n_iter)
            logger.log("computing loss after")
            logger.log("optimization finished")
