"""First order optimizer."""
import time

from dowel import logger
import pyprind
import tensorflow as tf

from garage.np.optimizers import BatchDataset
from garage.tf.misc import tensor_utils
from garage.tf.optimizers.utils import LazyDict


class FirstOrderOptimizer:
    """First order optimier.

    Performs (stochastic) gradient descent, possibly using fancier methods like
    ADAM etc.

    Args:
        tf_optimizer_cls (tf.Optimizer): Optimizer to be used.
        tf_optimizer_args (dict): Optimizer arguments.
        max_epochs (int): Maximum number of epochs for update.
        tolerance (float): Tolerance for difference in loss during update.
        batch_size (int): Batch size for optimization.
        callback (callable): Function to call during each epoch. Default is
            None.
        verbose (bool): If true, intermediate log message will be printed.
        name (str): Name scope of the optimizer.

    """

    def __init__(self,
                 tf_optimizer_cls=None,
                 tf_optimizer_args=None,
                 max_epochs=1000,
                 tolerance=1e-6,
                 batch_size=32,
                 callback=None,
                 verbose=False,
                 name='FirstOrderOptimizer'):
        self._opt_fun = None
        self._target = None
        self._callback = callback
        if tf_optimizer_cls is None:
            tf_optimizer_cls = tf.compat.v1.train.AdamOptimizer
        if tf_optimizer_args is None:
            tf_optimizer_args = dict(learning_rate=1e-3)
        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._verbose = verbose
        self._input_vars = None
        self._train_op = None
        self._name = name

    # pylint: disable=unused-argument
    def update_opt(self, loss, target, inputs, extra_inputs=None, **kwargs):
        """Construct operation graph for the optimizer.

        Args:
            loss (tf.Tensor): Loss objective to minimize.
            target (object): Target object to optimize. The object should
                implemenet `get_params()` and `get_param_values`.
            inputs (list[tf.Tensor]): List of input placeholders.
            extra_inputs (list[tf.Tensor]): List of extra input placeholders.
            kwargs (dict): Extra unused keyword arguments. Some optimizers
                have extra input, e.g. KL constraint.

        """
        with tf.name_scope(
                self._name,
                values=[loss, target.get_params(), inputs, extra_inputs]):

            self._target = target

            self._train_op = self._tf_optimizer.minimize(
                loss, var_list=target.get_params())

            if extra_inputs is None:
                extra_inputs = list()
            self._input_vars = inputs + extra_inputs
            self._opt_fun = LazyDict(
                f_loss=lambda: tensor_utils.compile_function(
                    inputs + extra_inputs, loss), )

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
            extra_inputs = tuple()
        return self._opt_fun['f_loss'](*(tuple(inputs) + extra_inputs))

        # pylint: disable=too-many-branches
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

        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log('Epoch {}'.format(epoch))
                progbar = pyprind.ProgBar(len(inputs[0]))

            for batch in dataset.iterate(update=True):
                sess.run(self._train_op,
                         dict(list(zip(self._input_vars, batch))))
                if self._verbose:
                    progbar.update(len(batch[0]))

            if self._verbose:
                if progbar.active:
                    progbar.stop()

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

            if abs(last_loss - new_loss) < self._tolerance:
                break
            last_loss = new_loss

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = self.__dict__.copy()
        del new_dict['_opt_fun']
        del new_dict['_tf_optimizer']
        del new_dict['_train_op']
        del new_dict['_input_vars']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        obj = type(self)()
        self.__dict__.update(obj.__dict__)
        self.__dict__.update(state)
