import copy


class Summary:
    def __init__(self, summary=None):
        self.summary = summary if summary else {}

    def reset(self):
        self.summary = {}

    def scalar(self, key, value):
        self.summary[key] = value

    def histogram(self, key, value):
        raise NotImplementedError

    def text(self, key, value):
        self.summary[key] = value

    def merge(self, other):
        raise NotImplementedError

    def to_dict(self):
        return self.summary.copy()

    def copy(self):
        return copy.copy(self)

    def __str__(self):
        return self.summary.__str__()
