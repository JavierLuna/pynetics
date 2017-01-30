import pickle
from abc import ABCMeta
from tempfile import TemporaryFile
from unittest import TestCase

from pynetics import Population
from pynetics.exceptions import InvalidSizeError
from pynetics.selections import BestIndividual
from test.utils import ConstantDiversity, DummySpawningPool, ConstantFitness


class SelectionTestCase(TestCase, metaclass=ABCMeta):
    """ Common configuration for all selection operator tests. """

    def setUp(self):
        self.sp_04 = DummySpawningPool(
            fitness=ConstantFitness(0.4),
            name='test1_0.4',
        )


class BestIndividualTestCase(SelectionTestCase):
    """ Tests for BestIndividual selection method. """

    def setUp(self):
        super().setUp()
        self.selection = BestIndividual()

    def test_class_is_pickeable(self):
        with TemporaryFile() as f:
            pickle.dump(self.selection, f)

    def test_when_population_size_is_lower_than_selection_size(self):
        p_size = 10
        population = Population(
            size=p_size,
            spawning_pool=self.sp_04,
            diversity=ConstantDiversity(0.5),
        )

        with self.assertRaises(InvalidSizeError):
            self.selection(population=population, n=p_size * 2)

    def test_when_population_size_is_equals_to_selection_size(self):
        p_size = 3

        population = Population(
            size=p_size,
            spawning_pool=self.sp_04,
            diversity=ConstantDiversity(0.5),
        )
        selected = self.selection(population=population, n=p_size)
        for i in population:
            self.assertIn(i, selected)

    def test_when_population_size_is_bigger_than_selection_size(self):
        p_size = 3

        population = Population(
            size=p_size,
            spawning_pool=self.sp_04,
            diversity=ConstantDiversity(0.5),
        )
        selected = self.selection(population=population, n=p_size - 1)
        for i in selected:
            self.assertIn(i, population)
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                self.assertFalse(selected[i] is selected[j])
