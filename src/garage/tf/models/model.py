"""Base model classes."""
import abc
from collections import namedtuple
import warnings

import tensorflow as tf

from garage.tf.models.module import Module


class BaseModel(abc.ABC):
    """Interface-only abstract class for models.

    A Model contains the structure/configuration of a set of computation
    graphs, or can be understood as a set of networks. Using a model
    requires calling `build()` with given input placeholder, which can
    be either tf.compat.v1.placeholder, or the output from another model. This
    makes composition of complex models with simple models much easier.

    Examples:
        model = SimpleModel(output_dim=2)
        # To use a model, first create a placeholder.
        # In the case of TensorFlow, we create a tf.compat.v1.placeholder.
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))

        # Building the model
        output = model.build(input_ph)

        # We can also pass the output of a model to another model.
        # Here we pass the output from the above SimpleModel object.
        model_2 = ComplexModel(output_dim=2)
        output_2 = model_2.build(output)

    """

    def build(self, *inputs, name=None):
        """Output of model with the given input placeholder(s).

        This function is implemented by subclasses to create their computation
        graphs, which will be managed by Model. Generally, subclasses should
        implement `build()` directly.

        Args:
            inputs (object): Input(s) for the model.
            name (str): Name of the model.

        Return:
            list[tf.Tensor]: Output(s) of the model.

        """

    @property
    @abc.abstractmethod
    def name(self):
        """Name for this Model."""

    @property
    @abc.abstractmethod
    def parameters(self):
        """Parameters of the Model.

        The output of a model is determined by its parameter. It could be
        the weights of a neural network model or parameters of a loss
        function model.

        Returns:
            list[tf.Tensor]: Parameters.

        """

    @parameters.setter
    def parameters(self, parameters):
        """Set parameters of the Model.

        Args:
            parameters (list[tf.Tensor]): Parameters.

        """


class Network:
    """Network class For TensorFlow.

    A Network contains connectivity information by inputs/outputs.
    When a Network is built, it appears as a subgraph in the computation
    graphs, scoped by the Network name. All Networks built with the same
    model share the same parameters, i.e same inputs yield to same outputs.
    """

    def __init__(self):
        self._inputs = None
        self._outputs = None

    @property
    def input(self):
        """Tensor input of the Network.

        Returns:
            tf.Tensor: Input.

        """
        return self._inputs[0]

    @property
    def inputs(self):
        """Tensor inputs of the Network.

        Returns:
            list[tf.Tensor]: Inputs.

        """
        return self._inputs

    @property
    def output(self):
        """Tensor output of the Network.

        Returns:
            tf.Tensor: Output.

        """
        return self._outputs[0]

    @property
    def outputs(self):
        """Tensor outputs of the Network.

        Returns:
            list[tf.Tensor]: Outputs.

        """
        return self._outputs


class Model(BaseModel, Module):
    r"""Model class for TensorFlow.

    A TfModel only contains the structure/configuration of the underlying
    computation graphs. Connectivity information are all in Network class.
    A TfModel contains zero or more Network.

    When a Network is created, it reuses the parameter from the
    model. If a Network is built without given a name, the name "default" will
    be used.

    ***
    Do not call tf.global_variable_initializers() after building a model as it
    will reassign random weights to the model.
    The parameters inside a model will be initialized when calling build().
    ***

    Pickling is handled automatcailly. The target weights should be assigned to
    self._default_parameters before pickling, so that the newly created model
    can check if target weights exist or not. When unpickled, the unserialized
    model will load the weights from self._default_parameters.

    The design is illustrated as the following:

         input_1                      input_2
            |                            |
    ============== Model (TfModel)===================
    |       |                            |          |
    |       |            Parameters      |          |
    |    =============  /           \  ============ |
    |    |  default  | /             \ | Network2 | |
    |    | (Network) |/               \|(Network) | |
    |    =============                 ============ |
    |       |                            |          |
    =================================================
            |                            |
            |                            |
    (outputs from 'default' networks)    |
                        outputs from ['Network2'] network


    Examples are also available in tests/garage/tf/models/test_model.

    Args:
      name (str): Name of the model. It will also become the variable scope
          of the model. Every model should have a unique name.

    """

    def __init__(self, name):
        super().__init__(name or type(self).__name__)
        self._networks = {}
        self._default_parameters = None
        self._variable_scope = None

    # pylint: disable=protected-access, assignment-from-no-return
    def build(self, *inputs, name=None):
        """Build a Network with the given input(s).

        ***
        Do not call tf.global_variable_initializers() after building a model
        as it will reassign random weights to the model.
        The parameters inside a model will be initialized when calling build().
        ***

        It uses the same, fixed variable scope for all Networks, to ensure
        parameter sharing. Different Networks must have an unique name.

        Args:
            inputs (list[tf.Tensor]) : Tensor input(s), recommended to be
                positional arguments, for example,
                def build(self, state_input, action_input, name=None).
            name (str): Name of the model, which is also the name scope of the
                model.

        Raises:
            ValueError: When a Network with the same name is already built.

        Returns:
            list[tf.Tensor]: Output tensors of the model with the given
                inputs.

        """
        network_name = name or 'default'
        if not self._networks:
            # First time building the model, so self._networks are empty
            # We store the variable_scope to reenter later when we reuse it
            with tf.compat.v1.variable_scope(self._name) as vs:
                self._variable_scope = vs
                with tf.name_scope(name=network_name):
                    network = Network()
                    network._inputs = inputs
                    network._outputs = self._build(*inputs, name)
                variables = self._get_variables().values()
                tf.compat.v1.get_default_session().run(
                    tf.compat.v1.variables_initializer(variables))
                if self._default_parameters:
                    self.parameters = self._default_parameters
        else:
            if network_name in self._networks:
                raise ValueError(
                    'Network {} already exists!'.format(network_name))
            with tf.compat.v1.variable_scope(self._variable_scope,
                                             reuse=True,
                                             auxiliary_name_scope=False):
                with tf.name_scope(name=network_name):
                    network = Network()
                    network._inputs = inputs
                    network._outputs = self._build(*inputs, name)
        custom_in_spec = self.network_input_spec()
        custom_out_spec = self.network_output_spec()
        in_spec = ['input', 'inputs']
        out_spec = ['output', 'outputs']
        in_args = [network.input, network.inputs]
        out_args = [network.output, network.outputs]
        if isinstance(network.inputs, tuple) and len(network.inputs) > 1:
            assert len(custom_in_spec) == len(network.inputs), (
                'network_input_spec must have same length as inputs!')
            in_spec.extend(custom_in_spec)
            in_args.extend(network.inputs)
        if isinstance(network.outputs, tuple) and len(network.outputs) > 1:
            assert len(custom_out_spec) == len(network.outputs), (
                'network_output_spec must have same length as outputs!')
            out_spec.extend(custom_out_spec)
            out_args.extend(network.outputs)
        elif len(custom_out_spec) > 0:
            if not isinstance(network.outputs, tuple):
                assert len(custom_out_spec) == 1, (
                    'network_input_spec must have same length as outputs!')
                out_spec.extend(custom_out_spec)
                out_args.extend([network.outputs])
            else:
                assert len(custom_out_spec) == len(network.outputs), (
                    'network_input_spec must have same length as outputs!')
                out_spec.extend(custom_out_spec)
                out_args.extend(network.outputs)

        c = namedtuple(network_name, [*in_spec, *out_spec])
        all_args = in_args + out_args
        out_network = c(*all_args)
        self._networks[network_name] = out_network

        return out_network

    def _build(self, *inputs, name=None):
        """Build this model given input placeholder(s).

        User should implement _build() inside their subclassed model,
        and construct the computation graphs in this function.

        Args:
            inputs: Tensor input(s), recommended to be position arguments, e.g.
                def _build(self, state_input, action_input, name=None).
                It would be usually same as the inputs in build().
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.

        Return:
            list[tf.Tensor]: Tensor output(s) of the model.

        """

    # pylint: disable=no-self-use
    def network_input_spec(self):
        """Network input spec.

        Return:
            list[str]: List of key(str) for the network inputs.

        """
        return []

    # pylint: disable=no-self-use
    def network_output_spec(self):
        """Network output spec.

        Return:
            list[str]: List of key(str) for the network outputs.

        """
        return []

    @property
    def parameters(self):
        """Parameters of the model.

        Returns:
            np.ndarray: Parameters

        """
        _variables = self._get_variables()
        if _variables:
            return tf.compat.v1.get_default_session().run(_variables)
        else:
            return _variables

    @parameters.setter
    def parameters(self, parameters):
        """Set model parameters.

        Args:
            parameters (tf.Tensor): Parameters.

        """
        variables = self._get_variables()
        for name, var in variables.items():
            found = False
            # param name without model name
            param_name = name[name.find(self.name) + len(self.name) + 1:]
            for k, v in parameters.items():
                if param_name in k:
                    var.load(v)
                    found = True
                    continue
            if not found:
                warnings.warn('No value provided for variable {}'.format(name))

    @property
    def name(self):
        """Name (str) of the model.

        This is also the variable scope of the model.

        Returns:
            str: Name of the model.

        """
        return self._name

    @property
    def input(self):
        """Default input of the model.

        When the model is built the first time, by default it
        creates the 'default' network. This property creates
        a reference to the input of the network.

        Returns:
            tf.Tensor: Default input of the model.

        """
        return self._networks['default'].input

    @property
    def output(self):
        """Default output of the model.

        When the model is built the first time, by default it
        creates the 'default' network. This property creates
        a reference to the output of the network.

        Returns:
            tf.Tensor: Default output of the model.

        """
        return self._networks['default'].output

    @property
    def inputs(self):
        """Default inputs of the model.

        When the model is built the first time, by default it
        creates the 'default' network. This property creates
        a reference to the inputs of the network.

        Returns:
            list[tf.Tensor]: Default inputs of the model.

        """
        return self._networks['default'].inputs

    @property
    def outputs(self):
        """Default outputs of the model.

        When the model is built the first time, by default it
        creates the 'default' network. This property creates
        a reference to the outputs of the network.

        Returns:
            list[tf.Tensor]: Default outputs of the model.

        """
        return self._networks['default'].outputs

    def _get_variables(self):
        """Get variables of this model.

        Returns:
            dict[str: tf.Tensor]: Variables of this model.

        """
        if self._variable_scope:
            return {v.name: v for v in self._variable_scope.global_variables()}
        else:
            return dict()

    def __getstate__(self):
        """Get the pickle state.

        Returns:
            dict: The pickled state.

        """
        new_dict = super().__getstate__()
        del new_dict['_networks']
        new_dict['_default_parameters'] = self.parameters
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): unpickled state.

        """
        super().__setstate__(state)
        self._networks = {}
