"""First order optimizer wrapped by PCGrad optimizer."""
import time

import click
from dowel import logger
import numpy as np
import tensorflow as tf

from garage import make_optimizer
from garage.np.optimizers import BatchDataset
from garage.tf.misc import tensor_utils
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.optimizers.utils import LazyDict

GATE_OP = 1


class PCGrad(tf.compat.v1.train.Optimizer):
    """Tensorflow implementation of PCGrad.

    Gradient Surgery for Multi-Task Learning. This the implementation from
    tianheyu927 (https://github.com/tianheyu927/PCGrad). The paper is
    https://arxiv.org/pdf/2001.06782.pdf.

    Args:
        optimizer (tf.Optimizer): The optimizer being wrapped.
        use_locking (bool): If True apply use locks to prevent concurrent
            updates to variables.
        name (str): The name to use for accumulators created for the optimizer.

    """

    def __init__(self, optimizer, use_locking=False, name='PCGrad'):
        super(PCGrad, self).__init__(use_locking, name)
        self.optimizer = optimizer

    def compute_gradients(self,
                          loss,
                          var_list=None,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        """Compute gradients of `loss` for the variables in `var_list`.

        Args:
            loss (list[tf.Tensor]): A list of Tensor containing the values to
                minimize.
            var_list (list[tf.Variable]): Optional list or tuple of
                `tf.Variable` to update to minimize `loss`. Defaults to the
                list of variables collected in the graph under the key
                `GraphKeys.TRAINABLE_VARIABLES`.
            gate_gradients (int): Not applicable.
            aggregation_method (tf.AggregationMethod): Not applicable.
            colocate_gradients_with_ops (bool): Not applicable.
            grad_loss (tf.Tensor): Optional. Not applicable.

        Returns:
            list: A list of (gradient, variable) pairs. Variable is always
                present, but gradient can be `None`.

        """
        assert isinstance(loss, list)
        num_tasks = len(loss)
        loss = tf.stack(loss)
        tf.random.shuffle(loss)

        # Compute per-task gradients.
        grads_task = tf.map_fn(
            lambda x: tf.concat([
                tf.reshape(grad, [
                    -1,
                ]) for grad in tf.gradients(x, var_list) if grad is not None
            ], 0), loss)

        # Compute gradient projections.
        def proj_grad(grad_task):
            """Project gradients of a task.

            Args:
                grad_task (tf.Tensor): Gradients to be projected.

            Returns:
                tf.Tensor: projected gradients.

            """
            for k in range(num_tasks):
                inner_product = tf.reduce_sum(grad_task * grads_task[k])
                proj_direction = inner_product / tf.reduce_sum(
                    grads_task[k] * grads_task[k])
                grad_task = grad_task - tf.minimum(proj_direction,
                                                   0.) * grads_task[k]
            return grad_task

        proj_grads_flatten = tf.map_fn(proj_grad, grads_task)

        # Unpack flattened projected gradients back to their original shapes.
        proj_grads = []
        for j in range(num_tasks):
            start_idx = 0
            for idx, var in enumerate(var_list):
                grad_shape = var.get_shape()
                flatten_dim = np.prod([
                    grad_shape.dims[i].value
                    for i in range(len(grad_shape.dims))
                ])
                proj_grad = proj_grads_flatten[j][start_idx:start_idx +
                                                  flatten_dim]
                proj_grad = tf.reshape(proj_grad, grad_shape)
                if len(proj_grads) < len(var_list):
                    proj_grads.append(proj_grad)
                else:
                    proj_grads[idx] += proj_grad
                start_idx += flatten_dim
        grads_and_vars = list(zip(proj_grads, var_list))
        return grads_and_vars

    # pylint: disable=protected-access
    # pylint: disable=missing-return-doc, missing-return-type-doc
    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)

    def _prepare(self):
        self.optimizer._prepare()

    def _apply_dense(self, grad, var):
        return self.optimizer._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, handle):
        return self.optimizer._resource_apply_dense(grad, handle)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        return self.optimizer._apply_sparse_shared(grad, var, indices,
                                                   scatter_add)

    def _apply_sparse(self, grad, var):
        return self.optimizer._apply_sparse(grad, var)

    def _resource_scatter_add(self, x, i, v):
        return self.optimizer._resource_scatter_add(x, i, v)

    def _resource_apply_sparse(self, grad, handle, indices):
        return self.optimizer._resource_apply_sparse(grad, handle, indices)

    def _finish(self, update_ops, name_scope):
        return self.optimizer._finish(update_ops, name_scope)

    def _call_if_callable(self, param):
        return self.optimizer._call_if_callable(param)

    # pylint: enable=protected-access
    # pylint: enable=missing-return-doc, missing-return-type-doc


class PCGradOptimizer(FirstOrderOptimizer):
    """PCGrad optimizer based on first order optimizer.

    Performs gradient surgery for Multi-Task Learning, possibly using fancier
    methods like ADAM etc.

    Args:
        optimizer (tf.Optimizer): Optimizer to be used.
        learning_rate (dict): learning rate arguments.
            learning rates are our main interest parameters to tune optimizers.
        max_episode_length (int): Maximum number of epochs for update.
        tolerance (float): Tolerance for difference in loss during update.
        batch_size (int): Batch size for optimization.
        callback (callable): Function to call during each epoch. Default is
            None.
        verbose (bool): If true, intermediate log message will be printed.
        use_locking (bool): If True apply use locks to prevent concurrent
            updates to variables.
        name (str): Name scope of the optimizer.

    """

    def __init__(self,
                 optimizer=None,
                 learning_rate=None,
                 max_episode_length=1000,
                 tolerance=1e-6,
                 batch_size=32,
                 callback=None,
                 verbose=False,
                 use_locking=False,
                 name='PCGradOptimizer'):
        self._use_locking = dict(use_locking=use_locking)
        super().__init__(optimizer=optimizer,
                         learning_rate=learning_rate,
                         max_episode_length=max_episode_length,
                         tolerance=tolerance,
                         batch_size=batch_size,
                         callback=callback,
                         verbose=verbose,
                         name=name)

    def update_opt(self, loss, target, inputs, extra_inputs=None, **kwargs):
        """Construct operation graph for the optimizer.

        Args:
            loss (list[tf.Tensor]): List of loss objectives to minimize.
            target (object): Target object to optimize. The object should
                implement `get_params()` and `get_param_values`.
            inputs (list[tf.Tensor]): List of input placeholders.
            extra_inputs (list[tf.Tensor]): List of extra input placeholders.
            kwargs (dict): Extra unused keyword arguments. Some optimizers
                have extra input, e.g. KL constraint.

        """
        del kwargs
        with tf.name_scope(self._name):
            self._target = target
            tf_optimizer = PCGrad(
                make_optimizer(self._tf_optimizer, **self._learning_rate,
                               **self._use_locking))
            self._train_op = tf_optimizer.minimize(
                loss, var_list=target.get_params())

            if extra_inputs is None:
                extra_inputs = list()
            self._input_vars = inputs + extra_inputs
            self._opt_fun = LazyDict(
                f_loss=lambda: tensor_utils.compile_function(
                    inputs + extra_inputs, loss), )

    def optimize(self, inputs, extra_inputs=None, callback=None):
        """Perform optimization.

        Args:
            inputs (list[numpy.ndarray]): List of input values.
            extra_inputs (list[numpy.ndarray]): List of extra input values.
            callback (callable): Function to call during each epoch. Default
                is None.

        Raises:
            NotImplementedError: If inputs are invalid.
            Exception: If loss function is None, i.e. not defined.

        """
        if not inputs:
            # Assumes that we should always sample mini-batches
            raise NotImplementedError('No inputs are fed to optimizer.')
        if self._opt_fun is None:
            raise Exception(
                'Use update_opt() to setup the loss function first.')

        f_loss = self._opt_fun['f_loss']

        if extra_inputs is None:
            extra_inputs = tuple()

        last_loss = f_loss(*(tuple(inputs) + extra_inputs))

        start_time = time.time()

        dataset = BatchDataset(inputs,
                               self._batch_size,
                               extra_inputs=extra_inputs)

        sess = tf.compat.v1.get_default_session()

        for epoch in range(self._max_episode_length):
            if self._verbose:
                logger.log('Epoch {}'.format(epoch))

            with click.progressbar(length=len(inputs[0]),
                                   label='Optimizing minibatches') as pbar:
                for batch in dataset.iterate(update=True):
                    sess.run(self._train_op,
                             dict(list(zip(self._input_vars, batch))))

                    pbar.update(len(batch[0]))

            new_loss = f_loss(*(tuple(inputs) + extra_inputs))

            if self._verbose:
                logger.log('Epoch: {} | Loss: {}'.format(epoch, new_loss))
            if self._callback or callback:
                elapsed = time.time() - start_time
                callback_args = dict(
                    loss=new_loss,
                    params=self._target.get_param_values()
                    if self._target else None,
                    itr=epoch,
                    elapsed=elapsed,
                )
                if self._callback:
                    self._callback(callback_args)
                if callback:
                    callback(**callback_args)

            if abs(np.mean(last_loss) - np.mean(new_loss)) < self._tolerance:
                break
            last_loss = new_loss
