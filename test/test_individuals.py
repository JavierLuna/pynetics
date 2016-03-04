import pickle
import unittest

from tempfile import TemporaryFile

from test import dummies


class IndividualTestCase(unittest.TestCase):
    """ Tests for Individual instances. """

    def test_individual_is_pickeable(self):
        """ Checks the individual by writing it in a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(dummies.DummyIndividual(), f)


class SpawningPoolTestCase(unittest.TestCase):
    """ Tests for SpawningPool instances. """

    def test_spawning_pool_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(dummies.DummySpawningPool(
                fitness=dummies.DummyFitness()
            ), f)
