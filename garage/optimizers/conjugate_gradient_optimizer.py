import numpy as np
import theano
import theano.tensor as TT

from garage.core import Serializable
from garage.misc import ext
from garage.misc import krylov
from garage.misc import logger
from garage.misc.ext import sliced_fun
from garage.theano.misc import tensor_utils


class PerlmutterHvp(Serializable):
    def __init__(self, num_slices=1):
        Serializable.quick_init(self, locals())
        self.target = None
        self.reg_coeff = None
        self.opt_fun = None
        self._num_slices = num_slices

    def update_opt(self, f, target, inputs, reg_coeff):
        self.target = target
        self.reg_coeff = reg_coeff
        params = target.get_params(trainable=True)

        constraint_grads = theano.grad(
            f, wrt=params, disconnected_inputs='warn')
        xs = tuple([tensor_utils.new_tensor_like("%s x" % p.name, p) \
                    for p in params])

        def hx_plain():
            hx_plain_splits = TT.grad(
                TT.sum([TT.sum(g * x) for g, x in zip(constraint_grads, xs)]),
                wrt=params,
                disconnected_inputs='warn')
            return TT.concatenate([TT.flatten(s) for s in hx_plain_splits])

        self.opt_fun = ext.LazyDict(
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


class FiniteDifferenceHvp(Serializable):
    def __init__(self,
                 base_eps=1e-8,
                 symmetric=True,
                 grad_clip=None,
                 num_slices=1):
        Serializable.quick_init(self, locals())
        self.base_eps = base_eps
        self.symmetric = symmetric
        self.grad_clip = grad_clip
        self._num_slices = num_slices

    def update_opt(self, f, target, inputs, reg_coeff):
        self.target = target
        self.reg_coeff = reg_coeff

        params = target.get_params(trainable=True)

        constraint_grads = theano.grad(
            f, wrt=params, disconnected_inputs='warn')
        flat_grad = tensor_utils.flatten_tensor_variables(constraint_grads)

        def f_hx_plain(*args):
            inputs_ = args[:len(inputs)]
            xs = args[len(inputs):]
            flat_xs = np.concatenate([np.reshape(x, (-1, )) for x in xs])
            param_val = self.target.get_param_values(trainable=True)
            eps = np.cast['float32'](
                self.base_eps / (np.linalg.norm(param_val) + 1e-8))
            self.target.set_param_values(
                param_val + eps * flat_xs, trainable=True)
            flat_grad_dvplus = self.opt_fun["f_grad"](*inputs_)
            if self.symmetric:
                self.target.set_param_values(
                    param_val - eps * flat_xs, trainable=True)
                flat_grad_dvminus = self.opt_fun["f_grad"](*inputs_)
                hx = (flat_grad_dvplus - flat_grad_dvminus) / (2 * eps)
                self.target.set_param_values(param_val, trainable=True)
            else:
                self.target.set_param_values(param_val, trainable=True)
                flat_grad = self.opt_fun["f_grad"](*inputs_)
                hx = (flat_grad_dvplus - flat_grad) / eps
            return hx

        self.opt_fun = ext.LazyDict(
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
    """
    Performs constrained optimization via line search. The search direction is
    computed using a conjugate gradient algorithm, which gives x = A^{-1}g,
    where A is a second order approximation of the constraint and g is the
    gradient of the loss function.
    """

    def __init__(self,
                 cg_iters=10,
                 reg_coeff=1e-5,
                 subsample_factor=1.,
                 backtrack_ratio=0.8,
                 max_backtracks=15,
                 accept_violation=False,
                 hvp_approach=None,
                 num_slices=1):
        """

        :param cg_iters: The number of CG iterations used to calculate A^-1 g
        :param reg_coeff: A small value so that A -> A + reg*I
        :param subsample_factor: Subsampling factor to reduce samples when
         using "conjugate gradient. Since the computation time for the descent
         direction dominates, this can greatly reduce the overall computation
         time.
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

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None
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
                   constraint_name="constraint",
                   *args,
                   **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should
         implement methods of the
         :class:`garage.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon),
         of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs, which could be
         subsampled if needed. It is assumed that the first dimension of these
         inputs should correspond to the number of data points
        :param extra_inputs: A list of symbolic variables as extra inputs which
         should not be subsampled
        :return: No return value.
        """

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)

        constraint_term, constraint_value = leq_constraint

        params = target.get_params(trainable=True)
        grads = theano.grad(loss, wrt=params, disconnected_inputs='warn')
        flat_grad = tensor_utils.flatten_tensor_variables(grads)

        self._hvp_approach.update_opt(
            f=constraint_term,
            target=target,
            inputs=inputs + extra_inputs,
            reg_coeff=self._reg_coeff)

        self._target = target
        self._max_constraint_val = constraint_value
        self._constraint_name = constraint_name

        self._opt_fun = ext.LazyDict(
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
                 subsample_grouped_inputs=None):

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
                subsample_inputs += tuple([x[inds] for x in inputs_grouped])
        else:
            subsample_inputs = inputs

        logger.log("computing loss before")
        loss_before = sliced_fun(self._opt_fun["f_loss"],
                                 self._num_slices)(inputs, extra_inputs)
        logger.log("performing update")
        logger.log("computing descent direction")

        flat_g = sliced_fun(self._opt_fun["f_grad"],
                            self._num_slices)(inputs, extra_inputs)

        hx = self._hvp_approach.build_eval(subsample_inputs + extra_inputs)

        descent_direction = krylov.cg(hx, flat_g, cg_iters=self._cg_iters)

        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val *
            (1. / (descent_direction.dot(hx(descent_direction)) + 1e-8)))
        if np.isnan(initial_step_size):
            initial_step_size = 1.
        flat_descent_step = initial_step_size * descent_direction

        logger.log("descent direction computed")

        prev_param = np.copy(self._target.get_param_values(trainable=True))
        n_iter = 0
        for n_iter, ratio in enumerate(self._backtrack_ratio**np.arange(
                self._max_backtracks)):  # yapf: disable
            cur_step = ratio * flat_descent_step
            cur_param = prev_param - cur_step
            self._target.set_param_values(cur_param, trainable=True)
            loss, constraint_val = sliced_fun(
                self._opt_fun["f_loss_constraint"],
                self._num_slices)(inputs, extra_inputs)
            if loss < loss_before \
               and constraint_val <= self._max_constraint_val:
                break
        if (np.isnan(loss) or np.isnan(constraint_val) or loss >= loss_before
                or constraint_val >= self._max_constraint_val) \
           and not self._accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
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
