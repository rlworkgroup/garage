"""Environment step data type."""
import collections


def Step(observation, reward, done, **kwargs):  # noqa: N802
    """Create a namedtuple from the results of environment.step(action).

    Provides the option to put extra diagnostic info in the kwargs (if it
    exists) without demanding an explicit positional argument.

    Args:
        observation (object): Agent's observation of the current environment
        reward (float) : Amount of reward returned after previous action
        done (bool): Whether the episode has ended, in which case further
            step() calls will return undefined results
        kwargs: Keyword args

    Returns:
        collections.namedtuple: A named tuple of the arguments.

    """
    return _Step(observation, reward, done, kwargs)


_Step = collections.namedtuple('Step',
                               ['observation', 'reward', 'done', 'info'])
