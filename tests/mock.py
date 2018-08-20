from unittest import mock


class PickleableMagicMock(mock.MagicMock):
    def __reduce__(self):
        return (mock.MagicMock, ())
