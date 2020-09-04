"""Data types for TensorFlow optimizers."""


class LazyDict:
    """An immutable, lazily-evaluated dict.

    Args:
        **kwargs (dict[Callable]): Initial lazy key-value pairs.
    """

    def __init__(self, **kwargs):
        self._lazy_dict = kwargs
        self._dict = {}

    def __getitem__(self, key):
        """See :meth:`object.__getitem__`.

        Args:
           key (Hashable): Key associated with the value to retrieve.

        Returns:
            object: Lazily-evaluated value of the :class:`Callable` associated
               with key.

        """
        if key not in self._dict:
            self._dict[key] = self._lazy_dict[key]()
        return self._dict[key]

    def __setitem__(self, key, value):
        """See :meth:`object.__setitem__`.

        Args:
            key (Hashable): Key associated with value.
            value (Callable): Function which returns the lazy value associated
                with key.

        """
        self.set(key, value)

    def get(self, key, default=None):
        """See :meth:`dict.get`.

        Args:
            key (Hashable): Key associated with the value to retreive.
            default (object): Value to return if key is not present in this
               :class:`LazyDict`.

        Returns:
            object: Value associated with key if the key is present, otherwise
                default.
        """
        if key in self._lazy_dict:
            return self[key]

        return default

    def set(self, key, value):
        """See :meth:`dict.set`.

        Args:
            key (Hashable): Key associated with value.
            value (Callable): Function which returns the lazy value associated
                with key.

        """
        self._lazy_dict[key] = value
