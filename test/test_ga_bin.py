import pickle
from tempfile import TemporaryFile
from unittest import TestCase

from pynetics.ga_bin import BinaryIndividualSpawningPool
from pynetics.ga_bin import GeneralizedRecombination


class BinaryIndividualSpawningPoolTestCase(TestCase):
    """ Tests for instances of this class. """

    def test_class_is_pickeable(self):
        """ Checks if it's pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(BinaryIndividualSpawningPool(10), f)


class GeneralizedRecombinationTestCase(TestCase):
    def test_class_is_pickeable(self):
        """ Checks if it's pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(GeneralizedRecombination(), f)
