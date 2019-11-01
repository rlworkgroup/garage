import time

from dowel import logger
import pyprind
import tensorflow as tf

from garage.np.optimizers import BatchDataset
from garage.tf.misc import tensor_utils
from garage.tf.optimizers.utils import LazyDict


class FirstOrderOptimizer:
    """
    Performs (stochastic) gradient descent, possibly using fancier methods like
    ADAM etc.
    """

    def __init__(
            self,
            tf_optimizer_cls=None,
            tf_optimizer_args=None,
            # learning_rate=1e-3,
            max_epochs=1000,
            tolerance=1e-6,
            batch_size=32,
            callback=None,
            verbose=False,
            name='FirstOrderOptimizer',
            **kwargs):
        """
        :param max_epochs:
        :param tolerance:
        :param update_method:
        :param batch_size: None or an integer. If None the whole dataset will
         be used.
        :param callback:
        :param kwargs:
        :return:
        """
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

    def update_opt(self, loss, target, inputs, extra_inputs=None, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should
         implement methods of the
        :class:`garage.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon),
         of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """
        with tf.name_scope(self._name,
                           values=[
                               loss,
                               target.get_params(trainable=True), inputs,
                               extra_inputs
                           ]):

            self._target = target

            self._train_op = self._tf_optimizer.minimize(
                loss, var_list=target.get_params(trainable=True))

            # updates = OrderedDict(
            #     [(k, v.astype(k.dtype)) for k, v in updates.iteritems()])

            if extra_inputs is None:
                extra_inputs = list()
            self._input_vars = inputs + extra_inputs
            self._opt_fun = LazyDict(
                f_loss=lambda: tensor_utils.compile_function(
                    inputs + extra_inputs, loss), )

    def loss(self, inputs, extra_inputs=None):
        if self._opt_fun is None:
            raise Exception(
                'Use update_opt() to setup the loss function first.')
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun['f_loss'](*(tuple(inputs) + extra_inputs))

    def optimize(self, inputs, extra_inputs=None, callback=None):

        if not inputs:
            # Assumes that we should always sample mini-batches
            raise NotImplementedError
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
                    params=self._target.get_param_values(
                        trainable=True) if self._target else None,
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
        """Object.__getstate__."""
        new_dict = self.__dict__.copy()
        del new_dict['_opt_fun']
        del new_dict['_tf_optimizer']
        del new_dict['_train_op']
        del new_dict['_input_vars']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        obj = type(self)()
        self.__dict__.update(obj.__dict__)
        self.__dict__.update(state)
