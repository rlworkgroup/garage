"""Utility functions used by tensorflow algorithms."""


class _Default:  # pylint: disable=too-few-public-methods
    """A wrapper class to represent default arguments.

    Args:
        val (object): Argument value.

    """

    def __init__(self, val):
        self.val = val


def make_optimizer(optimizer_type, **kwargs):
    """Create an optimizer for tensorflow algos.

    Args:
        optimizer_type (type): Type of optimizer.
        kwargs (dict): Other keyword arguments to initialize optimizer. This
            is not used when `optimizer_type` is tuple.

    Returns:
        tf.compat.v1.train.Optimizer: Constructed optimizer.
    """
    opt_args = {}
    for name, arg in kwargs.items():
        if isinstance(arg, _Default):
            opt_args[name] = arg.val
        else:
            opt_args[name] = arg
    return optimizer_type(**opt_args)
