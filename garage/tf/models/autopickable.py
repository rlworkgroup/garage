"""
The file contains classes for pickling tf.keras.models.Model.

This base class pickle a tf.keras.models.Model using get_config() and
get_weights() from tf.keras modules.
"""
import tensorflow as tf


class PickleCall:
    """
    PickleCall.

    This class tells pickler what function and arguments to call when
    unpickled. Instances are created using obj = C.__new__(C, *args)
    where args is the result of calling __getnewargs__()
    on the original object, when unpickled.

    Args:
        cls: Target class.
        to_call: Function to call when unpickled. In this case, the function
            is build_layers, which essentially tells pickler to build the
            layer. It will replace __pickle_target when unpickled.
        args: Arguments to the function call.
        kwargs: Keyword arguments to the function call.
        __pickle_target: The function to be called when unpickled. In this
            case, None when constructed the first time, and will be replaced by
            to_call afterwards.
    """

    def __new__(cls, to_call, args=(), kwargs=None, __pickle_target=None):
        """object.__new__."""
        if __pickle_target is None:
            result = super(cls, PickleCall).__new__(cls)
            result.to_call = to_call
            result.args = args
            result.kwargs = kwargs or {}
            return result
        else:
            # when unpickling
            return __pickle_target(*args, **kwargs)

    def __getnewargs__(self):
        """object.__getnewargs__."""
        return (None, self.args, self.kwargs, self.to_call)


def build_layers(config, weights, **kwargs):
    """Build a tf.keras.layer.Layer."""
    model = tf.keras.models.Model.from_config(config, custom_objects=kwargs)
    model.set_weights(weights)
    return model


class AutoPickable:
    """
    AutoPickable.

    Using PickleCall class to pickle tf.keras.model.Model.
    """

    def __getstate__(self):
        """object.__getstate__."""
        state = self.__dict__.copy()
        for k, v in self.__dict__.items():
            if isinstance(v, tf.keras.models.Model):
                custom_objects = {}
                for c in state[k].layers:
                    if "garage" in str(type(c)):  # detect subclassed layer
                        name = type(c).__name__
                        custom_objects[name] = type(c)
                state[k] = PickleCall(build_layers,
                                      (v.get_config(), v.get_weights()),
                                      custom_objects)
        return state
