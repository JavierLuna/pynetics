import random
from typing import TypeVar


def take_chances(probability: float = 0.5) -> bool:
    """ Given a probability, the method generates a random value to see if is
        lower or not than that probability.

    :param probability: The value of the probability to beat. Default is 0.5.
    :return: A value of True if the value geneated is bellow the probability
        specified, and false otherwise.
    """
    return random.random() < probability


T = TypeVar('T')


def clone_empty(obj: T) -> T:
    """ Used by classes which need to be cloned avoiding the call to __init__.

    :param obj: The object to be cloned.
    :return: A newly empty object of the class obj.
    """

    class Empty(obj.__class__):
        def __init__(self): pass

    empty: T = Empty()
    empty.__class__ = obj.__class__
    return empty
