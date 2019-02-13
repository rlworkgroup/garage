"""Base model classes."""
import abc
import warnings

import tensorflow as tf


class BaseModel(abc.ABC):
    """Interface-only abstract class for models."""

    def build(self, *inputs):
        """
        Output of model with the given input placeholder(s).

        This function is implemented by subclasses to create their computation
        graphs, which will be managed by Model. Generally, subclasses should
        implement `build()` directly.
        """
        pass

    @property
    def name(self):
        """Name for this Model."""
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
    def parameters(self):
        """Parameters of the Model."""
        pass

    @parameters.setter
    def parameters(self, parameters):
        """Set parameters of the Model."""
        pass


class AutoPickableModel(BaseModel, metaclass=abc.ABCMeta):
    """
    Abstract class for models with automatic pickling.

    This class follows the BaseModel API and handles pickling automatically.
    It provides both self.init_spec and self.parameters for pickling, which
    are the arguments and weights of the Model. When unpickled, a new model
    will be reconstructed with self.init_spec as arguments and load the weights
    from self.parameters.

    The target weights should be assigned to self._default_parameters during
    pickling, so that the newly created model can check if target weights exist
    or not.
    """

    def __init__(self):
        self._name = type(self).__name__  # name default to class
        self._inputs = None
        self._outputs = None
        self._default_parameters = None

    @property
    def init_spec(self):
        """
        Model specification.

        Positional and named arguments (args and kwargs) which, when passed
        to `__init__`, will construct this model.
        """
        pass

    @property
    def name(self):
        """Name of the model."""
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

    @property
    def parameters(self):
        """Parameters of the Model."""
        raise NotImplementedError

    @parameters.setter
    def parameters(self, parameters):
        """Set model parameters."""
        raise NotImplementedError

    def __getstate__(self):
        """Object.__getstate__."""
        args, kwargs = self.init_spec
        return {'args': args, 'kwargs': kwargs, 'parameters': self.parameters}

    def __setstate__(self, data):
        """Object.__setstate__."""
        # TODO(krzentner): There has got to be a better way.
        out = type(self)(*data['args'], **data['kwargs'])
        out._default_parameters = data['parameters']
        self.__dict__.update(out.__dict__)


class TfModel(AutoPickableModel):
    r"""
    Model class for TensorFlow.

    This class is a parent-child design.
    The parent model contains the parameters, i.e. the structure, and
    two child model are constructed with input_1 and input_2. Inputs and
    outputs of the parent model will be binded with the default child, which
    is the first child constructed.

    When new child model is created, it reuses the parameter from the
    parent model and can be accessed by calling model.child_2, where
    child_2 is the name given when building this particular child model.
    If a child is constructed without given a name, the name "default" will
    be used.

    The design is illustrated as the following:

         input_1                      input_2
            |                            |
    ============== Parent (TfModel)===============
    |       |                            |       |
    |       |          Parameters        |       |
    |    ===========  /           \  =========== |
    |    | Child 1 | /             \ | Child 2 | |
    |    |(TfModel)|/               \|(TfModel)| |
    |    ===========                 =========== |
    |       |                            |       |
    ==============================================
            |                            |
       (model.outputs) OR                |
    (model.default.outputs)        model.child_2.outputs

    The parameters inside a model will be initialized when calling build().
    Do not call tf.global_variable_initializers() after building a model as it
    will reassign random weights to the model.

    Examples are also available in tests/garage/tf/models/test_new_model.

    Args:
      name (str): Name of the model. It will also become the variable scope
      of the model. Every model should have a unique name.
    """

    def __init__(self, name):
        super().__init__()
        self._name = name or self._name
        self._is_parent = True
        self._is_built = False
        self._variable_scope = tf.variable_scope(self.name, reuse=False)
        self._name_scope = tf.name_scope("default")
        self._siblings = {}

    @classmethod
    def _new_child(cls, other, name):
        """Class method for constructing child model with the given name."""
        twin = cls(**other.init_spec[1])
        twin._name = name
        twin._is_parent = False
        twin._default_parameters = other._default_parameters
        twin._is_built = other._is_built
        twin._variable_scope = other.variable_scope
        twin._name_scope = tf.name_scope(name)
        return twin

    def build(self, *inputs, name=None):
        """
        Build a child model with the given input(s).

        ***Do not call tf.global_variable_initializers() after building a model
        as it will reassign random weights to the model.***

        It uses the fixed variable scope for all child models, to ensure
        parameter sharing. Different child models should have unique name.

        Only the parent model can be built. This ensures the model tree to only
        have two level: the parent and the children. Similarly, only the parent
        model can be pickled. Connectivity does not hold in pickling. The model
        needs to be built again afterwards.

        Args:
          name(str): Name of the model, which is also the name scope of the
            model.

        Raises:
          ValueError when
            - A child model with the same name exists
            - A child model is built.
            - A parent model wit the same name exists.

        Returns:
          outputs: Output tensor of the model with the given inputs.

        """
        if not name:
            if 'default' in self._siblings:
                raise ValueError("Subgraph {} already exists!".format(
                    self.name))
            with self._name_scope:
                if not self._is_built:
                    _variable_scope = self.variable_scope
                else:
                    _variable_scope = tf.variable_scope(
                        self._variable_scope._name_or_scope, reuse=True)
                with _variable_scope:
                    self._inputs = inputs
                    outputs = self._build(*inputs)
                    self._outputs = outputs
                    self._siblings['default'] = self
                    if not self._is_built:
                        variables = self._get_variables().values()
                        tf.get_default_session().run(
                            tf.variables_initializer(variables))
                        if self._default_parameters:
                            self.parameters = self._default_parameters
                        self._is_built = True
        elif name in self._siblings:
            raise ValueError("Subgraph {} already exists!".format(name))
        elif not self._is_parent:
            raise ValueError("Child of a model should not be built!")
        else:
            twin = type(self)._new_child(self, name)
            setattr(self, name, twin)
            self._siblings[name] = twin
            outputs = twin.build(*inputs)
            if not self._is_built:
                self._inputs = inputs
                self._outputs = list(outputs)
                self._is_built = True

        return outputs

    def _build(self, *inputs):
        """
        Output of the model given input placeholder(s).

        User should implement _build inside their subclassed model.
        """
        pass

    @property
    def variable_scope(self):
        """Variable scope of the model."""
        return self._variable_scope

    @property
    def name_scope(self):
        """Name scope of the model."""
        return self._name_scope

    @property
    def parameters(self):
        """Parameters of the model."""
        return tf.get_default_session().run(self._get_variables())

    @parameters.setter
    def parameters(self, parameters):
        """Set model parameters."""
        variables = self._get_variables()
        for name, var in variables.items():
            if name in parameters:
                var.load(parameters[name])
            else:
                warnings.warn('No value provided for variable {}'.format(name))

    def _get_variables(self):
        return {
            v.name: v
            for v in tf.global_variables(
                scope=self.variable_scope._name_or_scope)
        }

    def __getstate__(self):
        """Object.__getstate__."""
        if not self._is_parent:
            raise ValueError("Child of a model should not be pickled!")

        return super().__getstate__()
