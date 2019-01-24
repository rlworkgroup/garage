import abc


def get_now_timestamp():
    import datetime
    return datetime.datetime.now().isoformat()


def get_timestamp(name):
    import re
    isoformat_regex = r'\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d\.\d+'
    return re.search(isoformat_regex, name).group(0)


def cat_for_fname(*args):
    return '_'.join(args) + '.pkl'


class Checkpointer(abc.ABC):
    def __init__(self, prefix, resume=True):
        self.resume = resume
        self.prefix = prefix

    @abc.abstractmethod
    def load(self, **kwargs):
        """Load from or initialize checkpoint.

        Args:
            **kwargs: named objects to save if no checkpoint exists.

        Returns:
            dict: named objects loaded from checkpoint.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, **kwargs):
        """Save objects to a new checkpoint.

        Args:
            **kwargs: named objects to save.

        """
        raise NotImplementedError
