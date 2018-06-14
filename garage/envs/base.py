import collections

from cached_property import cached_property

from garage.envs import EnvSpec

_Step = collections.namedtuple("Step",
                               ["observation", "reward", "done", "info"])


def Step(observation, reward, done, **kwargs):
    """
    Convenience method creating a namedtuple with the results of the
    environment.step method.
    Put extra diagnostic info in the kwargs
    """
    return _Step(observation, reward, done, kwargs)
