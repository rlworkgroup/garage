"""This file contains the abstraction class for models."""
import abc
import collections
import warnings

import tensorflow as tf

from garage.tf.models.auto_pickable import PickleCall
from garage.tf.models import AutoPickable


class BaseModel(abc.ABC):
    """Interface-only abstract class for models."""
    @property
    def init_spec(self):
        """Positional and named arguments (args and kwargs) which, when passed
        to `__init__`, will construct this model"""
        pass

    def build(self, *inputs):
        """Output of model with the given input placeholder(s)."""
        pass

    @property
    def name(self):
        """A unique name for this Model."""
        pass

    @property
    def input(self):
        """Tensor input of the Model."""
        pass

    @property
    def output(self):
        """Tensor output of the Model."""
        pass

    @property
    def inputs(self):
        """Tensor inputs of the Model."""
        pass

    @property
    def outputs(self):
        """Tensor outputs of the Model."""
        pass

    @property
    def weights(self):
        """Weights of the Model."""
        pass

    @weights.setter
    def weights(self, weights):
        """Set weights of the Model."""
        pass


class Model(BaseModel, metaclass=abc.ABCMeta):
    """Abstract class for models with sane defaults."""
    def __init__(self):
        self._name = type(self).__name__  # name default to class
        self._inputs = None
        self._outputs = None
        self._default_weights = None
        self._built = False

    def build(self, *inputs):
        """Output of model with the given input placeholder(s)."""

        # Generally, you should not be rebuilding models
        if not self._built:
            self._inputs = inputs
            self._outputs = self._build(*self.inputs)
            self._built = True
        else:
            # TODO: This should probably only be a warning right now to help
            # with refactoring the current code. Eventually it should be an
            # exception.
            warnings.warn('Rebuilding model {}.'.format(self._name))

        return self.outputs

    def _build(self, *inputs):
        """Output of the model given input placeholder(s).

        This function is implemented by subclasses to create their computation
        graphs, which will be managed by Model. Generally, subclasses should not
        implement `build()` directly.
        """
        pass

    @property
    def name(self):
        return self._name

    @property
    def input(self):
        """Tensor input of the Model."""
        return self._inputs[0]

    @property
    def output(self):
        """Tensor output of the Model."""
        return self._outputs[0]

    @property
    def inputs(self):
        """Tensor inputs of the Model."""
        return self._inputs

    @property
    def outputs(self):
        """Tensor outputs of the Model."""
        return self._outputs

    def __getstate__(self):
        args, kwargs = self.init_spec
        return {
            'args': args,
            'kwargs': kwargs,
            'weights': self.weights
        }

    def __setstate__(self, data):
        # TODO(krzentner): There has got to be a better way.
        out = type(self)(*data['args'], **data['kwargs'])
        out._default_weights = data['weights']
        self.__dict__.update(out.__dict__)


class TfModel(Model):

    def __init__(self, name):
        super().__init__()
        # All TF models need a valid unique name
        # TODO(ahtsan): produce a message on collision?
        self._name = name or self._name
        # All ops and variables for this Model *MUST* be placed under this scope
        self._scope = tf.variable_scope(self.name, reuse=False)

    def build(self, *inputs):
        with self._scope:
            super().build(*inputs)
        # Initialize our variables so that they're guaranteed to exist after
        # build
        #
        # NOTE: It's very important that we don't run
        # `tf.global_variable_initializer()` after this step (which will call
        # the mostly-random initializers on each variable, destroying the
        # presets loaded from the file)
        # @gautams has already hit this problem in his own code.
        #
        # TODO(ahtsan): We can deal with this by sanitizing
        # `tf.global_variable_initializer()` from the codebase and only
        # initializing variables as-needed. This would also be a good
        # opportunity to create a process-global session manager so we can stop
        # relying on `tf.get_default_session()` everywhere.
        #
        # TODO(ahtsan): It might be better if we could instead set the
        # initializers for these variables to
        # tf.constant_initializer(weight_value), but that would appear to
        # require monkeypatching tf.get_variable()
        # Doing this is not unheard-of:
        # https://github.com/deepmind/learning-to-learn/blob/master/meta.py#L80
        variables = self._get_variables().values()
        tf.get_default_session().run(tf.variables_initializer(variables))
        if self._default_weights:
            self.weights = self._default_weights

    @property
    def scope(self):
        return self._scope

    @property
    def weights(self):
        return tf.get_default_session().run(self._get_variables())

    @weights.setter
    def weights(self, weights):
        variables = self._get_variables()
        for name, var in variables.items():
            if name in weights:
                var.load(weights[name])
            else:
                warnings.warn('No value provided for variable {}'.format(name))

    def _get_variables(self):
        return {v.name: v for v in tf.global_variables(scope=self.name)}


class PickableModel(AutoPickable):
    """Abstraction class for autopickable models."""

    def _build_model(self, input_var):
        """
        Build model.

        This function will build the whole graph for
        the model. All tensors should be created here.
        By calling this function, a copy of the same graph
        should be created with the same parameters.
        """
        raise NotImplementedError

    def __call__(self, inputs):
        """Output of model with the given input placeholder."""
        return self.model(inputs)

    @property
    def input(self):
        """Tensor input of the Model."""
        return self.model.input

    @property
    def output(self):
        """Tensor output of the Model."""
        return self.model.output

    @property
    def inputs(self):
        """Tensor inputs of the Model."""
        return self.model.inputs

    @property
    def outputs(self):
        """Tensor outputs of the Model."""
        return self.model.outputs

    @property
    def dist(self):
        """
        Distribution of model.

        Raise NotImplementError when model does not have a distribution.
        """
        if 'distribution_layer' in [l.name for l in self.model.layers]:
            return self.model.get_layer('distribution_layer').dist
        else:
            raise NotImplementedError
