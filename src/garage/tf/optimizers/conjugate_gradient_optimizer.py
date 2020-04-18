"""Conjugate Gradient Optimizer.

Computes the decent direction using the conjugate gradient method, and then
computes the optimal step size that will satisfy the KL divergence constraint.
Finally, it performs a backtracking line search to optimize the objective.

"""
import abc

from dowel import logger
import numpy as np
import tensorflow as tf

from garage.tf.misc import tensor_utils
from garage.tf.optimizers.utils import LazyDict, sliced_fun


class HessianVectorProduct(abc.ABC):
    """Base class for computing Hessian-vector product.

    Args:
        num_slices (int): Hessian-vector product function's inputs will be
            divided into num_slices and then averaged together to improve
            performance.

    """

    def __init__(self, num_slices=1):
        self._target = None
        self._reg_coeff = None
        self._hvp_fun = None
        self._num_slices = num_slices

    @abc.abstractmethod
    def update_hvp(self, f, target, inputs, reg_coeff, name=None):
        """Build the symbolic graph to compute the Hessian-vector product.

        Args:
            f (tf.Tensor): The function whose Hessian needs to be computed.
            target (garage.tf.policies.Policy): A parameterized object to
                optimize over.
            inputs (tuple[tf.Tensor]): The inputs for function f.
            reg_coeff (float): A small value so that A -> A + reg*I.
            name (str): Name to be used in tf.name_scope.

        """

    def build_eval(self, inputs):
        """Build the evaluation function.  # noqa: D202, E501 # https://github.com/PyCQA/pydocstyle/pull/395.

        Args:
            inputs (tuple[numpy.ndarray]): Function f will be evaluated on
                these inputs.

        Returns:
            function: It can be called to get the final result.

        """

        def _eval(v):
            """The evaluation function.

            Args:
                v (numpy.ndarray): The vector to be multiplied with Hessian.

            Returns:
                numpy.ndarray: The product of Hessian of function f and v.

            """
            xs = tuple(self._target.flat_to_params(v))
            ret = sliced_fun(self._hvp_fun['f_hx_plain'], self._num_slices)(
                inputs, xs) + self._reg_coeff * v
            return ret

        return _eval

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = self.__dict__.copy()
        del new_dict['_hvp_fun']
        return new_dict


class PearlmutterHvp(HessianVectorProduct):
    """Computes Hessian-vector product using Pearlmutter's algorithm.

    `Pearlmutter, Barak A. "Fast exact multiplication by the Hessian." Neural
     computation 6.1 (1994): 147-160.`

    """

    def update_hvp(self, f, target, inputs, reg_coeff, name='PearlmutterHvp'):
        """Build the symbolic graph to compute the Hessian-vector product.

        Args:
            f (tf.Tensor): The function whose Hessian needs to be computed.
            target (garage.tf.policies.Policy): A parameterized object to
                optimize over.
            inputs (tuple[tf.Tensor]): The inputs for function f.
            reg_coeff (float): A small value so that A -> A + reg*I.
            name (str): Name to be used in tf.name_scope.

        """
        self._target = target
        self._reg_coeff = reg_coeff
        params = target.get_params()
        with tf.name_scope(name):
            constraint_grads = tf.gradients(f,
                                            xs=params,
                                            name='gradients_constraint')
            for idx, (grad, param) in enumerate(zip(constraint_grads, params)):
                if grad is None:
                    constraint_grads[idx] = tf.zeros_like(param)

            xs = tuple([
                tensor_utils.new_tensor_like(p.name.split(':')[0], p)
                for p in params
            ])

            def hx_plain():
                """Computes product of Hessian(f) and vector v.

                Returns:
                    tf.Tensor: Symbolic result.

                """
                with tf.name_scope('hx_plain'):
                    with tf.name_scope('hx_function'):
                        hx_f = tf.reduce_sum(
                            tf.stack([
                                tf.reduce_sum(g * x)
                                for g, x in zip(constraint_grads, xs)
                            ])),
                    hx_plain_splits = tf.gradients(hx_f,
                                                   params,
                                                   name='gradients_hx_plain')
                    for idx, (hx,
                              param) in enumerate(zip(hx_plain_splits,
                                                      params)):
                        if hx is None:
                            hx_plain_splits[idx] = tf.zeros_like(param)
                    return tensor_utils.flatten_tensor_variables(
                        hx_plain_splits)

            self._hvp_fun = LazyDict(
                f_hx_plain=lambda: tensor_utils.compile_function(
                    inputs=inputs + xs,
                    outputs=hx_plain(),
                    log_name='f_hx_plain',
                ), )


class FiniteDifferenceHvp(HessianVectorProduct):
    """Computes Hessian-vector product using finite difference method.

    Args:
        base_eps (float): Base epsilon value.
        symmetric (bool): Symmetric or not.
        num_slices (int): Hessian-vector product function's inputs will be
            divided into num_slices and then averaged together to improve
            performance.

    """

    def __init__(self, base_eps=1e-8, symmetric=True, num_slices=1):
        super().__init__(num_slices=num_slices)
        self.base_eps = base_eps
        self.symmetric = symmetric

    def update_hvp(self, f, target, inputs, reg_coeff, name='FiniteDifferenceHvp'):
        """Build the symbolic graph to compute the Hessian-vector product.

        Args:
            f (tf.Tensor): The function whose Hessian needs to be computed.
            target (garage.tf.policies.Policy): A parameterized object to
                optimize over.
            inputs (tuple[tf.Tensor]): The inputs for function f.
            reg_coeff (float): A small value so that A -> A + reg*I.
            name (str): Name to be used in tf.name_scope.

        """
        self._target = target
        self._reg_coeff = reg_coeff
        params = target.get_params()
        with tf.name_scope(name):
            constraint_grads = tf.gradients(f,
                                            xs=params,
                                            name='gradients_constraint')
            for idx, (grad, param) in enumerate(zip(constraint_grads, params)):
                if grad is None:
                    constraint_grads[idx] = tf.zeros_like(param)
            flat_grad = tensor_utils.flatten_tensor_variables(constraint_grads)

            def f_hx_plain(*args):
                """Computes product of Hessian(f) and vector v.

                Args:
                    args (tuple[numpy.ndarray]): Contains inputs of function f
                        , and vector v.

                Returns:
                    tf.Tensor: Symbolic result.

                """
                with tf.name_scope('f_hx_plain'):
                    inputs_ = args[:len(inputs)]
                    xs = args[len(inputs):]
                    flat_xs = np.concatenate(
                        [np.reshape(x, (-1, )) for x in xs])
                    param_val = self._target.get_param_values()
                    eps = np.cast['float32'](
                        self.base_eps / (np.linalg.norm(param_val) + 1e-8))
                    self._target.set_param_values(param_val + eps * flat_xs)
                    flat_grad_dvplus = self._hvp_fun['f_grad'](*inputs_)
                    self._target.set_param_values(param_val)
                    if self.symmetric:
                        self._target.set_param_values(param_val -
                                                      eps * flat_xs)
                        flat_grad_dvminus = self._hvp_fun['f_grad'](*inputs_)
                        hx = (flat_grad_dvplus - flat_grad_dvminus) / (2 * eps)
                        self._target.set_param_values(param_val)
                    else:
                        flat_grad = self._hvp_fun['f_grad'](*inputs_)
                        hx = (flat_grad_dvplus - flat_grad) / eps
                    return hx

            self._hvp_fun = LazyDict(
                f_grad=lambda: tensor_utils.compile_function(
                    inputs=inputs,
                    outputs=flat_grad,
                    log_name='f_grad',
                ),
                f_hx_plain=lambda: f_hx_plain,
            )


class ConjugateGradientOptimizer:
    """Performs constrained optimization via line search.

    The search direction is computed using a conjugate gradient algorithm,
    which gives x = A^{-1}g, where A is a second order approximation of the
    constraint and g is the gradient of the loss function.

    Args:
        cg_iters (int): The number of CG iterations used to calculate A^-1 g
        reg_coeff (float): A small value so that A -> A + reg*I
        subsample_factor (float): Subsampling factor to reduce samples when
            using "conjugate gradient. Since the computation time for the
            descent direction dominates, this can greatly reduce the overall
            computation time.
        backtrack_ratio (float): backtrack ratio for backtracking line search.
        max_backtracks (int): Max number of iterations for backtrack
            linesearch.
        accept_violation (bool): whether to accept the descent step if it
            violates the line search condition after exhausting all
            backtracking budgets.
        hvp_approach (HessianVectorProduct): A class that computes
            Hessian-vector products.
        num_slices (int): Hessian-vector product function's inputs will be
            divided into num_slices and then averaged together to improve
            performance.

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
            hvp_approach = PearlmutterHvp(num_slices)
        self._hvp_approach = hvp_approach

    def update_opt(
            self,
            loss,
            target,
            leq_constraint,
            inputs,
            extra_inputs=None,
            name='ConjugateGradientOptimizer',
            constraint_name='constraint',
    ):
        """Update the optimizer.

        Build the functions for computing loss, gradient, and
        the constraint value.

        Args:
            loss (tf.Tensor): Symbolic expression for the loss function.
            target (garage.tf.policies.Policy): A parameterized object to
                optimize over.
            leq_constraint (tuple[tf.Tensor, float]): A constraint provided
                as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
            inputs (list(tf.Tenosr)): A list of symbolic variables as inputs,
                which could be subsampled if needed. It is assumed that the
                first dimension of these inputs should correspond to the
                number of data points.
            extra_inputs (list[tf.Tenosr]): A list of symbolic variables as
                extra inputs which should not be subsampled.
            name (str): Name to be passed to tf.name_scope.
            constraint_name (str): A constraint name for prupose of logging
                and variable names.

        """
        params = target.get_params()
        ns_vals = [loss, target, leq_constraint, inputs, extra_inputs, params]
        with tf.name_scope(name):
            inputs = tuple(inputs)
            if extra_inputs is None:
                extra_inputs = tuple()
            else:
                extra_inputs = tuple(extra_inputs)

            constraint_term, constraint_value = leq_constraint

            with tf.name_scope('loss_gradients'):
                grads = tf.gradients(loss, xs=params)
                for idx, (grad, param) in enumerate(zip(grads, params)):
                    if grad is None:
                        grads[idx] = tf.zeros_like(param)
                flat_grad = tensor_utils.flatten_tensor_variables(grads)

            self._hvp_approach.update_hvp(f=constraint_term,
                                          target=target,
                                          inputs=inputs + extra_inputs,
                                          reg_coeff=self._reg_coeff,
                                          name='update_opt_' + constraint_name)

            self._target = target
            self._max_constraint_val = constraint_value
            self._constraint_name = constraint_name

            self._opt_fun = LazyDict(
                f_loss=lambda: tensor_utils.compile_function(
                    inputs=inputs + extra_inputs,
                    outputs=loss,
                    log_name='f_loss',
                ),
                f_grad=lambda: tensor_utils.compile_function(
                    inputs=inputs + extra_inputs,
                    outputs=flat_grad,
                    log_name='f_grad',
                ),
                f_constraint=lambda: tensor_utils.compile_function(
                    inputs=inputs + extra_inputs,
                    outputs=constraint_term,
                    log_name='constraint',
                ),
                f_loss_constraint=lambda: tensor_utils.compile_function(
                    inputs=inputs + extra_inputs,
                    outputs=[loss, constraint_term],
                    log_name='f_loss_constraint',
                ),
            )

    def loss(self, inputs, extra_inputs=None):
        """Compute the loss value.

        Args:
            inputs (list[numpy.ndarray]): A list inputs, which could be
                subsampled if needed. It is assumed that the first dimension
                of these inputs should correspond to the number of data points
            extra_inputs (list[numpy.ndarray]): A list of extra inputs which
                should not be subsampled.

        Returns:
            float: Loss value.

        """
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return sliced_fun(self._opt_fun['f_loss'],
                          self._num_slices)(inputs, extra_inputs)

    def constraint_val(self, inputs, extra_inputs=None):
        """Constraint value.

        Args:
            inputs (list[numpy.ndarray]): A list inputs, which could be
                subsampled if needed. It is assumed that the first dimension
                of these inputs should correspond to the number of data points
            extra_inputs (list[numpy.ndarray]): A list of extra inputs which
                should not be subsampled.

        Returns:
            float: Constraint value.

        """
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return sliced_fun(self._opt_fun['f_constraint'],
                          self._num_slices)(inputs, extra_inputs)

    def optimize(self,
                 inputs,
                 extra_inputs=None,
                 subsample_grouped_inputs=None,
                 name='optimize'):
        """Optimize the function.

        Args:
            inputs (list[numpy.ndarray]): A list inputs, which could be
                subsampled if needed. It is assumed that the first dimension
                of these inputs should correspond to the number of data points
            extra_inputs (list[numpy.ndarray]): A list of extra inputs which
                should not be subsampled.
            subsample_grouped_inputs (list[numpy.ndarray]): Subsampled inputs
                to be used when subsample_factor is less than one.
            name (str): The name argument for tf.name_scope.

        """
        with tf.name_scope(name):
            prev_param = np.copy(self._target.get_param_values())
            inputs = tuple(inputs)
            if extra_inputs is None:
                extra_inputs = tuple()

            subsample_inputs = inputs
            if self._subsample_factor < 1:
                if subsample_grouped_inputs is None:
                    subsample_grouped_inputs = [inputs]
                subsample_inputs = tuple()
                for inputs_grouped in subsample_grouped_inputs:
                    n_samples = len(inputs_grouped[0])
                    inds = np.random.choice(n_samples,
                                            int(n_samples *
                                                self._subsample_factor),
                                            replace=False)
                    subsample_inputs += tuple(
                        [x[inds] for x in inputs_grouped])

            logger.log(
                ('Start CG optimization: '
                 '#parameters: %d, #inputs: %d, #subsample_inputs: %d') %
                (len(prev_param), len(inputs[0]), len(subsample_inputs[0])))

            logger.log('computing loss before')
            loss_before = sliced_fun(self._opt_fun['f_loss'],
                                     self._num_slices)(inputs, extra_inputs)

            logger.log('computing gradient')
            flat_g = sliced_fun(self._opt_fun['f_grad'],
                                self._num_slices)(inputs, extra_inputs)
            logger.log('gradient computed')

            logger.log('computing descent direction')
            hx = self._hvp_approach.build_eval(subsample_inputs + extra_inputs)
            descent_direction = cg(hx, flat_g, cg_iters=self._cg_iters)

            initial_step_size = np.sqrt(
                2.0 * self._max_constraint_val *
                (1. / (descent_direction.dot(hx(descent_direction)) + 1e-8)))
            if np.isnan(initial_step_size):
                initial_step_size = 1.
            flat_descent_step = initial_step_size * descent_direction

            logger.log('descent direction computed')

            n_iter = 0
            for n_iter, ratio in enumerate(self._backtrack_ratio**np.arange(
                    self._max_backtracks)):  # yapf: disable
                cur_step = ratio * flat_descent_step
                cur_param = prev_param - cur_step
                self._target.set_param_values(cur_param)
                loss, constraint_val = sliced_fun(
                    self._opt_fun['f_loss_constraint'],
                    self._num_slices)(inputs, extra_inputs)
                if (loss < loss_before
                        and constraint_val <= self._max_constraint_val):
                    break
            if (np.isnan(loss) or np.isnan(constraint_val)
                    or loss >= loss_before or constraint_val >=
                    self._max_constraint_val) and not self._accept_violation:
                logger.log(
                    'Line search condition violated. Rejecting the step!')
                if np.isnan(loss):
                    logger.log('Violated because loss is NaN')
                if np.isnan(constraint_val):
                    logger.log('Violated because constraint %s is NaN' %
                               self._constraint_name)
                if loss >= loss_before:
                    logger.log('Violated because loss not improving')
                if constraint_val >= self._max_constraint_val:
                    logger.log('Violated because constraint %s is violated' %
                               self._constraint_name)
                self._target.set_param_values(prev_param)
            logger.log('backtrack iters: %d' % n_iter)
            logger.log('optimization finished')

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = self.__dict__.copy()
        del new_dict['_opt_fun']
        return new_dict


def cg(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    """Use Conjugate Gradient iteration to solve Ax = b. Demmel p 312.

    Args:
        f_Ax (function): A function to compute Hessian vector product.
        b (numpy.ndarray): Right hand side of the equation to solve.
        cg_iters (int): Number of iterations to run conjugate gradient
            algorithm.
        residual_tol (float): Tolerence for convergence.

    Returns:
        numpy.ndarray: Solution x* for equation Ax = b.

    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    for _ in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x
