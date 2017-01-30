import pickle
import unittest
from tempfile import TemporaryFile

from pynetics import Population
from pynetics.exceptions import RequiredValueError
from pynetics.replacements import LowElitism, HighElitism
from test.utils import DummySpawningPool, ConstantFitness, ConstantDiversity


class LowElitismTestCase(unittest.TestCase):
    """ Tests for low elitism replacement method. """

    def setUp(self):
        self.replacement = LowElitism()
        self.sp_04 = DummySpawningPool(
            fitness=ConstantFitness(0.4),
            name='test1_0.4',
        )
        self.sp_05 = DummySpawningPool(
            fitness=ConstantFitness(0.5),
            name='test1_0.5',
        )
        self.sp_06 = DummySpawningPool(
            fitness=ConstantFitness(0.6),
            name='test_0.6',
        )

    def test_class_is_pickeable(self):
        with TemporaryFile() as f:
            pickle.dump(self.replacement, f)

    def test_error_if_population_is_empty(self):
        with self.assertRaises(RequiredValueError):
            self.replacement(population=None, offspring=[])

    def test_no_individuals_remains_population_untouched(self):
        population = Population(
            size=10,
            spawning_pool=self.sp_04,
            diversity=ConstantDiversity(0.5),
        )
        population.sort()
        original = [i for i in population]
        self.replacement(population=population, offspring=[])
        population.sort()
        self.assertEqual(original, [i for i in population])

    def test_equal_length_replaces_all(self):
        population = Population(
            size=10,
            spawning_pool=self.sp_04,
            diversity=ConstantDiversity(0.5),
        )
        new_individuals = [self.sp_06.spawn() for _ in range(10)]
        self.replacement(
            population=population,
            offspring=new_individuals
        )
        population.sort()
        self.assertEqual(new_individuals, [i for i in population])

    def test_replacement_when_offspring_is_better(self):
        population = Population(
            size=10,
            spawning_pool=self.sp_06,
            diversity=ConstantDiversity(0.5),
            individuals=[self.sp_04.spawn() for _ in range(5)]
        )
        new_individuals = [self.sp_05.spawn() for _ in range(5)]
        self.replacement(
            population=population,
            offspring=new_individuals
        )
        self.assertEqual(0, len([i for i in population if i.fitness() == 0.4]))
        self.assertEqual(5, len([i for i in population if i.fitness() == 0.5]))
        self.assertEqual(5, len([i for i in population if i.fitness() == 0.6]))

    def test_replacement_when_offspring_is_worst(self):
        population = Population(
            size=10,
            spawning_pool=self.sp_06,
            diversity=ConstantDiversity(0.5),
            individuals=[self.sp_05.spawn() for _ in range(5)]
        )
        new_individuals = [self.sp_04.spawn() for _ in range(5)]
        self.replacement(
            population=population,
            offspring=new_individuals
        )
        self.assertEqual(5, len([i for i in population if i.fitness() == 0.4]))
        self.assertEqual(0, len([i for i in population if i.fitness() == 0.5]))
        self.assertEqual(5, len([i for i in population if i.fitness() == 0.6]))


class HighElitismTestCase(unittest.TestCase):
    """ Tests for high elitism replacement method. """

    def setUp(self):
        self.replacement = HighElitism()
        self.sp_04 = DummySpawningPool(
            fitness=ConstantFitness(0.4),
            name='test1_0.4',
        )
        self.sp_05 = DummySpawningPool(
            fitness=ConstantFitness(0.5),
            name='test1_0.5',
        )
        self.sp_06 = DummySpawningPool(
            fitness=ConstantFitness(0.6),
            name='test_0.6',
        )

    def test_class_is_pickeable(self):
        with TemporaryFile() as f:
            pickle.dump(self.replacement, f)

    def test_error_if_population_is_empty(self):
        with self.assertRaises(RequiredValueError):
            self.replacement(population=None, offspring=[])

    def test_no_individuals_remains_population_untouched(self):
        population = Population(
            size=10,
            spawning_pool=self.sp_04,
            diversity=ConstantDiversity(0.5),
        )
        population.sort()
        original = [i for i in population]
        self.replacement(population=population, offspring=[])
        population.sort()
        self.assertEqual(original, [i for i in population])

    def test_equal_length_replaces_all_when_offspring_is_better(self):
        population = Population(
            size=10,
            spawning_pool=self.sp_04,
            diversity=ConstantDiversity(0.5),
        )
        new_individuals = [self.sp_06.spawn() for _ in range(10)]
        self.replacement(
            population=population,
            offspring=new_individuals
        )
        population.sort()
        self.assertEqual(10, len([i for i in population if i.fitness() == 0.6]))

    def test_equal_length_replaces_nothing_when_offspring_is_worst(self):
        population = Population(
            size=10,
            spawning_pool=self.sp_06,
            diversity=ConstantDiversity(0.5),
        )
        new_individuals = [self.sp_04.spawn() for _ in range(10)]
        self.replacement(
            population=population,
            offspring=new_individuals
        )
        population.sort()
        self.assertEqual(10, len([i for i in population if i.fitness() == 0.6]))

    def test_replacement_when_offspring_is_better(self):
        population = Population(
            size=10,
            spawning_pool=self.sp_06,
            diversity=ConstantDiversity(0.5),
            individuals=[self.sp_04.spawn() for _ in range(5)]
        )
        new_individuals = [self.sp_05.spawn() for _ in range(5)]
        self.replacement(
            population=population,
            offspring=new_individuals
        )
        self.assertEqual(0, len([i for i in population if i.fitness() == 0.4]))
        self.assertEqual(5, len([i for i in population if i.fitness() == 0.5]))
        self.assertEqual(5, len([i for i in population if i.fitness() == 0.6]))

    def test_replacement_when_offspring_is_worst(self):
        population = Population(
            size=10,
            spawning_pool=self.sp_06,
            diversity=ConstantDiversity(0.5),
            individuals=[self.sp_05.spawn() for _ in range(5)]
        )
        new_individuals = [self.sp_04.spawn() for _ in range(5)]
        self.replacement(
            population=population,
            offspring=new_individuals
        )
        self.assertEqual(0, len([i for i in population if i.fitness() == 0.4]))
        self.assertEqual(5, len([i for i in population if i.fitness() == 0.5]))
        self.assertEqual(5, len([i for i in population if i.fitness() == 0.6]))
