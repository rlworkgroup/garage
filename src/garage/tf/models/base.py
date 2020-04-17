"""Base model classes."""
import abc
from collections import namedtuple
import warnings

import tensorflow as tf

from garage.misc.tensor_utils import flatten_tensors, unflatten_tensors


class BaseModel(abc.ABC):
    """Interface-only abstract class for models.

    A Model contains the structure/configuration of a set of computation
    graphs, or can be understood as a set of networks. Using a model
    requires calling `build()` with given input placeholder, which can be
    either tf.compat.v1.placeholder, or the output from another model. This
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


class Model(BaseModel):
    r"""Model class for TensorFlow.

    A TfModel only contains the structure/configuration of the underlying
    computation graphs. Connectivity information are all in Network class.
    A TfModel contains zero or more Network.

    When a Network is created, it reuses the parameter from the
    model and can be accessed by calling model.networks['network_name'],
    If a Network is built without given a name, the name "default" will
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
    (model.networks['default'].outputs)  |
                        model.networks['Network2'].outputs


    Examples are also available in tests/garage/tf/models/test_model.

    Args:
      name (str): Name of the model. It will also become the variable scope
          of the model. Every model should have a unique name.

    """

    def __init__(self, name):
        super().__init__()
        self._name = name or type(self).__name__  # name default to class
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

        c = namedtuple(network_name, [*in_spec, *out_spec])
        all_args = in_args + out_args
        self._networks[network_name] = c(*all_args)

        return network.outputs

    def _build(self, *inputs, name=None):
        """Output of the model given input placeholder(s).

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
    def networks(self):
        """Networks of the model.

        Returns:
            dict[str: Network]: Networks.

        """
        return self._networks

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
            if name in parameters:
                var.load(parameters[name])
            else:
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
        return self.networks['default'].input

    @property
    def output(self):
        """Default output of the model.

        When the model is built the first time, by default it
        creates the 'default' network. This property creates
        a reference to the output of the network.

        Returns:
            tf.Tensor: Default output of the model.

        """
        return self.networks['default'].output

    @property
    def inputs(self):
        """Default inputs of the model.

        When the model is built the first time, by default it
        creates the 'default' network. This property creates
        a reference to the inputs of the network.

        Returns:
            list[tf.Tensor]: Default inputs of the model.

        """
        return self.networks['default'].inputs

    @property
    def outputs(self):
        """Default outputs of the model.

        When the model is built the first time, by default it
        creates the 'default' network. This property creates
        a reference to the outputs of the network.

        Returns:
            list[tf.Tensor]: Default outputs of the model.

        """
        return self.networks['default'].outputs

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
        new_dict = self.__dict__.copy()
        del new_dict['_networks']
        new_dict['_default_parameters'] = self.parameters
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): unpickled state.

        """
        self.__dict__.update(state)
        self._networks = {}


class Module(abc.ABC):
    """A module that builds on top of model.

    Args:
        name (str): Module name, also the variable scope.

    """

    def __init__(self, name):
        self._name = name
        self._variable_scope = None
        self._cached_params = None
        self._cached_param_shapes = None

    @property
    def name(self):
        """str: Name of this module."""
        return self._name

    @property
    @abc.abstractmethod
    def vectorized(self):
        """bool: If this module supports vectorization input."""

    @property
    @abc.abstractmethod
    def recurrent(self):
        """bool: If this module has a hidden state."""

    def reset(self, do_resets=None):
        """Reset the module.

        This is effective only to recurrent modules. do_resets is effective
        only to vectoried modules.

        For a vectorized modules, do_resets is an array of boolean indicating
        which internal states to be reset. The length of do_resets should be
        equal to the length of inputs.

        Args:
            do_resets (numpy.ndarray): Bool array indicating which states
                to be reset.

        """

    @property
    def state_info_specs(self):
        """State info specification.

        Returns:
            List[str]: keys and shapes for the information related to the
                module's state when taking an action.

        """
        return list()

    @property
    def state_info_keys(self):
        """State info keys.

        Returns:
            List[str]: keys for the information related to the module's state
                when taking an input.

        """
        return [k for k, _ in self.state_info_specs]

    def terminate(self):
        """Clean up operation."""

    def get_trainable_vars(self):
        """Get trainable variables.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
                variable scope.

        """
        return self._variable_scope.trainable_variables()

    def get_global_vars(self):
        """Get global variables.

        Returns:
            List[tf.Variable]: A list of global variables in the current
                variable scope.

        """
        return self._variable_scope.global_variables()

    def get_params(self):
        """Get the trainable variables.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
                variable scope.

        """
        if self._cached_params is None:
            self._cached_params = self.get_trainable_vars()
        return self._cached_params

    def get_param_shapes(self):
        """Get parameter shapes.

        Returns:
            List[tuple]: A list of variable shapes.

        """
        if self._cached_param_shapes is None:
            params = self.get_params()
            param_values = tf.compat.v1.get_default_session().run(params)
            self._cached_param_shapes = [val.shape for val in param_values]
        return self._cached_param_shapes

    def get_param_values(self):
        """Get param values.

        Returns:
            np.ndarray: Values of the parameters evaluated in
                the current session

        """
        params = self.get_params()
        param_values = tf.compat.v1.get_default_session().run(params)
        return flatten_tensors(param_values)

    def set_param_values(self, param_values):
        """Set param values.

        Args:
            param_values (np.ndarray): A numpy array of parameter values.

        """
        param_values = unflatten_tensors(param_values, self.get_param_shapes())
        for param, value in zip(self.get_params(), param_values):
            param.load(value)

    def flat_to_params(self, flattened_params):
        """Unflatten tensors according to their respective shapes.

        Args:
            flattened_params (np.ndarray): A numpy array of flattened params.

        Returns:
            List[np.ndarray]: A list of parameters reshaped to the
                shapes specified.

        """
        return unflatten_tensors(flattened_params, self.get_param_shapes())

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = self.__dict__.copy()
        del new_dict['_cached_params']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        self._cached_params = None
        self.__dict__.update(state)


class StochasticModule(Module):
    """Stochastic Module."""

    @property
    @abc.abstractmethod
    def distribution(self):
        """Distribution."""

    @abc.abstractmethod
    def dist_info_sym(self,
                      input_var,
                      state_info_vars=None,
                      name='dist_info_sym'):
        """Symbolic graph of the distribution.

        Return the symbolic distribution information given input.

        Args:
            input_var (tf.Tensor): symbolic variable for input.
            state_info_vars (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the input.
            name (str): Name of the symbolic graph.

        """

    def dist_info(self, input_value, state_infos):
        """Distribution info.

        Return the distribution information given input.

        Args:
            input_value (tf.Tensor): Input values.
            state_infos (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the input.

        """
