"""Limited-memory BFGS (L-BFGS) optimizer."""
import time

import scipy.optimize
import tensorflow as tf

from garage.tf.misc import tensor_utils
from garage.tf.optimizers.utils import LazyDict


class LbfgsOptimizer:
    """Limited-memory BFGS (L-BFGS) optimizer.

    Performs unconstrained optimization via L-BFGS.

    Args:
        max_opt_itr (int): Maximum iteration for update.
        callback (callable): Function to call during optimization.

    """

    def __init__(self, max_opt_itr=20, callback=None):
        self._max_opt_itr = max_opt_itr
        self._opt_fun = None
        self._target = None
        self._callback = callback

    def update_opt(self,
                   loss,
                   target,
                   inputs,
                   extra_inputs=None,
                   name='LbfgsOptimizer',
                   **kwargs):
        """Construct operation graph for the optimizer.

        Args:
            loss (tf.Tensor): Loss objective to minimize.
            target (object): Target object to optimize. The object should
                implemenet `get_params()` and `get_param_values`.
            inputs (list[tf.Tensor]): List of input placeholders.
            extra_inputs (list[tf.Tensor]): List of extra input placeholders.
            name (str): Name scope.
            kwargs (dict): Extra unused keyword arguments. Some optimizers
                have extra input, e.g. KL constraint.

        """
        del kwargs
        self._target = target
        params = target.get_params()
        with tf.name_scope(name):

            def get_opt_output():
                """Helper function to construct graph.

                Returns:
                    list[tf.Tensor]: Loss and gradient tensor.

                """
                with tf.name_scope('get_opt_output'):
                    flat_grad = tensor_utils.flatten_tensor_variables(
                        tf.gradients(loss, params))
                    return [
                        tf.cast(loss, tf.float64),
                        tf.cast(flat_grad, tf.float64)
                    ]

            if extra_inputs is None:
                extra_inputs = list()

            self._opt_fun = LazyDict(
                f_loss=lambda: tensor_utils.compile_function(
                    inputs + extra_inputs, loss),
                f_opt=lambda: tensor_utils.compile_function(
                    inputs=inputs + extra_inputs,
                    outputs=get_opt_output(),
                ))

    def loss(self, inputs, extra_inputs=None):
        """The loss.

        Args:
            inputs (list[numpy.ndarray]): List of input values.
            extra_inputs (list[numpy.ndarray]): List of extra input values.

        Returns:
            float: Loss.

        Raises:
            Exception: If loss function is None, i.e. not defined.

        """
        if self._opt_fun is None:
            raise Exception(
                'Use update_opt() to setup the loss function first.')
        if extra_inputs is None:
            extra_inputs = list()
        return self._opt_fun['f_loss'](*(list(inputs) + list(extra_inputs)))

    def optimize(self, inputs, extra_inputs=None, name='optimize'):
        """Perform optimization.

        Args:
            inputs (list[numpy.ndarray]): List of input values.
            extra_inputs (list[numpy.ndarray]): List of extra input values.
            name (str): Name scope.

        Raises:
            Exception: If loss function is None, i.e. not defined.

        """
        if self._opt_fun is None:
            raise Exception(
                'Use update_opt() to setup the loss function first.')

        with tf.name_scope(name):
            f_opt = self._opt_fun['f_opt']

            if extra_inputs is None:
                extra_inputs = list()

            def f_opt_wrapper(flat_params):
                """Helper function to set parameters values.

                Args:
                    flat_params (numpy.ndarray): Flattened parameter values.

                Returns:
                    list[tf.Tensor]: Loss and gradient tensor.

                """
                self._target.set_param_values(flat_params)
                ret = f_opt(*inputs)
                return ret

            itr = [0]
            start_time = time.time()

            if self._callback:

                def opt_callback(params):
                    """Callback function wrapper.

                    Args:
                        params (numpy.ndarray): Parameters.

                    """
                    loss = self._opt_fun['f_loss'](*(inputs + extra_inputs))
                    elapsed = time.time() - start_time
                    self._callback(
                        dict(
                            loss=loss,
                            params=params,
                            itr=itr[0],
                            elapsed=elapsed,
                        ))
                    itr[0] += 1
            else:
                opt_callback = None

            scipy.optimize.fmin_l_bfgs_b(
                func=f_opt_wrapper,
                x0=self._target.get_param_values(),
                maxiter=self._max_opt_itr,
                callback=opt_callback,
            )

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = self.__dict__.copy()
        del new_dict['_opt_fun']
        return new_dict
