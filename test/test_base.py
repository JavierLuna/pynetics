import pickle
import unittest
from unittest.mock import Mock

from tempfile import TemporaryFile

from pynetics import PyneticsError, Population
from pynetics.exceptions import InvalidSize
from test import dummies


class StopConditionTestCase(unittest.TestCase):
    """ Tests for Individual instances. """

    def test_stop_condition_is_pickeable(self):
        """ Checks the individual by writing it in a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(dummies.DummyStopCondition(), f)


class IndividualTestCase(unittest.TestCase):
    """ Tests for Individual instances. """

    def test_individual_is_pickeable(self):
        """ Checks the individual by writing it in a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(dummies.DummyIndividual(), f)

    def test_computed_fitness_is_cached(self):
        individual = dummies.DummyIndividual()
        individual.fitness_method = dummies.DummyFitness()
        self.assertFalse(individual.cache_disabled)
        self.assertIsNone(individual.fitness_cached)
        individual.fitness()
        self.assertIsNotNone(individual.fitness_cached)

    def test_computed_init_fitness_is_not_cached(self):
        individual = dummies.DummyIndividual()
        individual.fitness_method = dummies.DummyFitness()
        self.assertFalse(individual.cache_disabled)
        self.assertIsNone(individual.fitness_cached)
        individual.fitness(init=True)
        self.assertIsNone(individual.fitness_cached)

    def test_computed_fitness_is_not_cached_if_cache_disabled(self):
        individual = dummies.DummyIndividual(disable_cache=True)
        individual.fitness_method = dummies.DummyFitness()
        self.assertTrue(individual.cache_disabled)
        self.assertIsNone(individual.fitness_cached)
        individual.fitness()
        self.assertIsNone(individual.fitness_cached)

    def test_computed_init_fitness_is_not_cached_if_cache_disabled(self):
        individual = dummies.DummyIndividual(disable_cache=True)
        individual.fitness_method = dummies.DummyFitness()
        self.assertTrue(individual.cache_disabled)
        self.assertIsNone(individual.fitness_cached)
        individual.fitness(init=True)
        self.assertIsNone(individual.fitness_cached)


class SpawningPoolTestCase(unittest.TestCase):
    """ Tests for SpawningPool instances. """

    def test_spawning_pool_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(dummies.DummySpawningPool(
                fitness=dummies.DummyFitness()
            ), f)

    def test_fitness_method_is_correctly_stored_after_initialization(self):
        """ If fitness method is correctly stored in the instance. """
        fitness = dummies.DummyFitness()
        spawning_pool = dummies.DummySpawningPool(fitness=fitness)
        self.assertIs(fitness, spawning_pool.fitness)

    def test_spawn_individual_assigns_the_fitness_method(self):
        fitness = dummies.DummyFitness()
        spawning_pool = dummies.DummySpawningPool(fitness=fitness)
        individual = spawning_pool.spawn()
        self.assertIs(fitness, individual.fitness_method)


class TestPopulation(unittest.TestCase):
    """ Test for populations. """

    # TODO Tests to check that the order of individuals is respected.

    @staticmethod
    def individuals(n=10, fitness_method=dummies.DummyFitness()):
        individuals = []
        for i in range(n):
            individual = dummies.DummyIndividual()
            individual.fitness_cached = i
            individual.fitness_method = fitness_method
            individuals.append(individual)
        return individuals

    def test_population_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(dummies.DummyPopulation(size=10), f)

    def test_cannot_initialize_a_population_of_size_0_or_less(self):
        for size in (-100, -10, -1, 0):
            with self.assertRaises(InvalidSize):
                dummies.DummyPopulation(size=size)

    def test_population_maintain_size_at_initialization_time(self):
        """ Size is kept when initializing.

        That means that, if a population is initializated with size n, it will
        have n individuals, regardless the number of individuals are passed as
        parameter.
        """
        size = 10
        individuals_list = [[], self.individuals(10), self.individuals(100)]
        for individuals in individuals_list:
            self.assertEquals(
                size,
                len(dummies.DummyPopulation(size=size, individuals=individuals))
            )

    def test_population_length_is_computed_correctly(self):
        sizes = (1, 10, 100, 1000)
        for size in sizes:
            self.assertEquals(size, len(dummies.DummyPopulation(size=size)))

    def test_population_shrinks_when_individual_is_removed(self):
        size = 10
        population = dummies.DummyPopulation(size=size)
        for i in range(10):
            self.assertEquals(size - i, len(population))
            del population[0]

    def test_individuals_are_correctly_added_to_the_population(self):
        size = 10
        individuals = self.individuals(size + 1)
        population = dummies.DummyPopulation(
            size=size,
            individuals=individuals[:10]
        )
        self.assertEquals(size, len(population))
        population.append(individuals[10])
        self.assertEquals(size + 1, len(population))
        self.assertIn(individuals[10], population)

    def test_population_is_sorted_when_accessing_individuals(self):
        population = dummies.DummyPopulation(size=10)
        self.assertTrue(population.sorted)
        for individual in self.individuals(10):
            population.append(individual)
            self.assertFalse(population.sorted)
            _ = population[0]
            self.assertTrue(population.sorted)

    def test_best_individuals_are_stored(self):
        size = 10
        population = dummies.DummyPopulation(size=size)
        population.genetic_algorithm = Mock()
        population.genetic_algorithm.generation = 0
        best_individuals = []
        for _ in range(100):
            population.evolve()
            population.genetic_algorithm.generation += 1
            best_individuals.append(population.best())

        self.assertListEqual(
            best_individuals,
            population.best_individuals_by_generation
        )

    def test_best_individuals_are_returned(self):
        size = 10
        individuals = self.individuals(size)
        population = dummies.DummyPopulation(size=size, individuals=individuals)
        self.assertEquals(
            population.best().fitness(),
            individuals[-1].fitness()
        )


class FitnessTestCase(unittest.TestCase):
    """ Tests for SpawningPool instances. """

    def test_fitness_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(dummies.DummyFitness(), f)


class MutationTestCase(unittest.TestCase):
    """ Tests for Mutation instances. """

    def test_mutation_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(dummies.DummyMutation(), f)


class RecombinationTestCase(unittest.TestCase):
    """ Tests for Recombination instances. """

    def test_mutation_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(dummies.DummyRecombination(), f)


class ReplacementTestCase(unittest.TestCase):
    """ Tests for Replacement instances. """

    def test_mutation_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(dummies.DummyReplacement(), f)


class SelectionTestCase(unittest.TestCase):
    """ Tests for Selection instances. """

    def test_mutation_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(dummies.DummySelection(), f)

    def test_repetition_defaults_to_false(self):
        """ Repetable parameter defaults to false. """
        self.assertFalse(dummies.DummySelection().repetable)

    def test_repetition_is_correctly_stored_after_initialization(self):
        """ Repetable parameter takes the value specified in initialization. """
        self.assertFalse(dummies.DummySelection(repetable=False).repetable)
        self.assertTrue(dummies.DummySelection(repetable=True).repetable)

    def test_error_if_more_selections_than_size_when_not_repetition(self):
        """ When no repetable and requested more individuals than available. """
        population = dummies.DummyPopulation(10)
        for i in range(1, 10):
            with self.assertRaises(PyneticsError):
                dummies.DummySelection()(population, len(population) + 1)


class CatastropheTestCase(unittest.TestCase):
    """ Tests for Catastrophe instances. """

    def test_mutation_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(dummies.DummyCatastrophe(), f)
