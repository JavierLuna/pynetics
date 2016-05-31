import pickle
import unittest
from tempfile import TemporaryFile

from pynetics.replacements import LowElitism


class LowElitismTestCase(unittest.TestCase):
    """ Tests for low elitism replacement method. """

    @staticmethod
    def test_class_is_pickeable():
        """ Checks if it's pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(LowElitism(), f)


class HighElitismTestCase(unittest.TestCase):
    """ Tests for low elitism replacement method. """

    @staticmethod
    def test_class_is_pickeable():
        """ Checks if it's pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(LowElitism(), f)
