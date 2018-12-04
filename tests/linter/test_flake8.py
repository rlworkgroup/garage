class BadClass():
    def __init__():  # E0213: Method should have "self" as first argument
        yield 1  # E0100: __init__ method is a generator
        return 0  # E0101: Explicit return in __init__
