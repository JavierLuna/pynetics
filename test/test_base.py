import pickle
import unittest

from tempfile import TemporaryFile
from unittest.mock import MagicMock

from pynetics import PyneticsError, GeneticAlgorithm, StopCondition
from pynetics.exceptions import InvalidSize
from test import utils


class MockGeneticAlgorithm(GeneticAlgorithm):
    def clone(self):
        return super().clone()

    def step(self):
        pass

    def best(self, generation=None):
        return True


class MockStopCondition(StopCondition):
    def __call__(self, genetic_algorithm):
        return True


class GeneticAlgorithmTestCase(unittest.TestCase):
    """ Tests for the GeneticAlgorithm super class. """

    def setUp(self):
        self.default_stop_condition = MockStopCondition()

    def test_instances_are_correctly_constructed(self):
        ga = MockGeneticAlgorithm(self.default_stop_condition)

        self.assertEquals(ga.stop_condition, self.default_stop_condition)
        self.assertEquals(ga.listeners, {})
        self.assertEquals(ga.generation, 0)

    def test_fluent_start_algorithm_clones_genetic_algorithm(self):
        ga_1 = MockGeneticAlgorithm(self.default_stop_condition)
        ga_2 = ga_1.on_start(lambda ga: print(ga.generation))

        self.assertIsNot(ga_1, ga_2)

    def test_fluent_end_algorithm_clones_genetic_algorithm(self):
        ga_1 = MockGeneticAlgorithm(self.default_stop_condition)
        ga_2 = ga_1.on_end(lambda ga: print(ga.generation))

        self.assertIsNot(ga_1, ga_2)

    def test_fluent_start_step_clones_genetic_algorithm(self):
        ga_1 = MockGeneticAlgorithm(self.default_stop_condition)
        ga_2 = ga_1.on_step_start(lambda ga: print(ga.generation))

        self.assertIsNot(ga_1, ga_2)

    def test_fluent_end_step_clones_genetic_algorithm(self):
        ga_1 = MockGeneticAlgorithm(self.default_stop_condition)
        ga_2 = ga_1.on_step_end(lambda ga: print(ga.generation))

        self.assertIsNot(ga_1, ga_2)


'''
    def run(self):
        """ Runs the simulation.

        The process is as follows: initialize populations and, while the stop
        condition is not met, do a new evolve step. This process relies in the
        abstract method "step".
        """
        self.initialize()
        self.call_listeners(GeneticAlgorithm.ALGORITHM_START)
        while self.best() is None or not self.stop_condition(self):
            self.call_listeners(GeneticAlgorithm.STEP_START)
            self.step()
            self.generation += 1
            self.call_listeners(GeneticAlgorithm.STEP_END)
        self.call_listeners(GeneticAlgorithm.ALGORITHM_END)
        self.finish()

    def call_listeners(self, message):
        [f(self) for f in self.listeners[message]]

    def initialize(self):
        """ Called when starting the genetic algorithm to initialize it. """
        self.generation = 0

    @staticmethod
    def finish():
        """ Called one the algorithm has finished. """
        pass

    @abstractmethod
    def step(self):
        """ Called on every iteration of the algorithm. """

    @abstractmethod
    def best(self, generation=None):
        """ Returns the best individual obtained until this moment.

        :param generation: The generation of the individual that we want to
            recover. If not set, this will be the one emerged in the last
            generation. Defaults to None (not set, thus last generation).
        :return: The best individual generated in the specified generation.
        """

    @abstractmethod
    def clone(self):
        """ Creates an instance as an exact copy of this algorithm.

        The implementing subclass must override this method calling the super
        class method because it has some attributes also to be cloned.

        :return: An exact copy of this genetic algorithm.
        """
        ga = clone_empty(self)
        ga.stop_condition = self.stop_condition
        ga.listeners = self.listeners
        ga.generation = self.generation
        return ga

    def on_start(self, f):
        """ Specifies a functor to be called when the algorithm starts.

        This function will be called AFTER initialization but BEFORE the first
        iteration, including the check against the stop condition.

        :param f: The functor to be called. It must accept a GeneticAlgorithm
            instance as a parameter.
        """
        ga = self.clone()
        ga.listeners[GeneticAlgorithm.ALGORITHM_START].append(f)
        return ga

    def on_end(self, f):
        """ Specifies a functor to be called when the algorithm ends.

        Particularly, this method will be called AFTER the stop condition
        has been met.

        :param f: The functor to be called. It must accept a GeneticAlgorithm
            instance as a parameter.
        """
        ga = self.clone()
        ga.listeners[GeneticAlgorithm.ALGORITHM_END].append(f)
        return ga

    def on_step_start(self, f):
        """ Specifies a functor to be called when an iteration step starts.

        This method will be called AFTER the stop condition has been checked
        and proved to be false) and BEFORE the new step is computed.

        :param f: The functor to be called. It must accept a GeneticAlgorithm
            instance as a parameter.
        """
        ga = self.clone()
        ga.listeners[GeneticAlgorithm.STEP_START].append(f)
        return ga

    def on_step_end(self, f):
        """ Specifies a functor to be called when an iteration ends.

        This method will be called AFTER an step of the algorithm has been
        computed and BEFORE a new check against the stop condition is going
        to be made.

        :param f: The functor to be called. It must accept a GeneticAlgorithm
            instance as a parameter.
        """
        ga = self.clone()
        ga.listeners[GeneticAlgorithm.STEP_END].append(f)
        return ga
'''


class StopConditionTestCase(unittest.TestCase):
    """ Tests for Individual instances. """

    def test_class_is_pickeable(self):
        """ Checks the individual by writing it in a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(utils.DummyStopCondition(), f)


class IndividualTestCase(unittest.TestCase):
    """ Tests for Individual instances. """

    def test_class_is_pickeable(self):
        """ Checks the individual by writing it in a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(utils.DummyIndividual(), f)


class SpawningPoolTestCase(unittest.TestCase):
    """ Tests for SpawningPool instances. """

    def test_class_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(utils.DummySpawningPool(), f)


class TestPopulation(unittest.TestCase):
    """ Test for populations. """

    def test_class_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(utils.DummyPopulation(size=10), f)

    def test_cannot_initialize_a_population_of_size_0_or_less(self):
        for size in (-100, -10, -1, 0):
            with self.assertRaises(InvalidSize):
                utils.DummyPopulation(size=size)

    def test_population_maintain_size_at_initialization_time(self):
        """ Size is kept when initializing.

        That means that, if a population is initializated with size n, it will
        have n individuals, regardless the number of individuals are passed as
        parameter.
        """
        size = 10
        individuals_list = [[], utils.individuals(10), utils.individuals(100)]
        for individuals in individuals_list:
            self.assertEquals(
                size,
                len(utils.DummyPopulation(size=size, individuals=individuals))
            )

    def test_population_length_is_computed_correctly(self):
        sizes = (1, 10, 100, 1000)
        for size in sizes:
            self.assertEquals(size, len(utils.DummyPopulation(size=size)))

    def test_population_shrinks_when_individual_is_removed(self):
        size = 10
        population = utils.DummyPopulation(size=size)
        for i in range(10):
            self.assertEquals(size - i, len(population))
            del population[0]

    def test_individuals_are_correctly_added_to_the_population(self):
        size = 10
        individuals = utils.individuals(size + 1)
        population = utils.DummyPopulation(
            size=size,
            individuals=individuals[:10]
        )
        self.assertEquals(size, len(population))
        population.append(individuals[10])
        self.assertEquals(size + 1, len(population))
        self.assertIn(individuals[10], population)

    def test_best_individuals_are_returned(self):
        size = 10
        individuals = utils.individuals(size)
        population = utils.DummyPopulation(size=size, individuals=individuals)
        self.assertEquals(
            population.best().fitness(),
            individuals[-1].fitness()
        )


class FitnessTestCase(unittest.TestCase):
    """ Tests for SpawningPool instances. """

    def test_class_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(utils.DummyFitness(), f)


class MutationTestCase(unittest.TestCase):
    """ Tests for Mutation instances. """

    def test_class_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(utils.DummyMutation(), f)


class RecombinationTestCase(unittest.TestCase):
    """ Tests for Recombination instances. """

    def test_class_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(utils.DummyRecombination(), f)


class ReplacementTestCase(unittest.TestCase):
    """ Tests for Replacement instances. """

    def test_class_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(utils.DummyReplacement(), f)


class SelectionTestCase(unittest.TestCase):
    """ Tests for Selection instances. """

    def test_class_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(utils.DummySelection(), f)

    def test_error_if_more_selections_than_size_when_not_repetition(self):
        """ When no repetable and requested more individuals than available. """
        population = utils.DummyPopulation(10)
        for i in range(1, 10):
            with self.assertRaises(PyneticsError):
                utils.DummySelection()(population, len(population) + 1)


class CatastropheTestCase(unittest.TestCase):
    """ Tests for Catastrophe instances. """

    def test_class_is_pickeable(self):
        """ Checks is pickeable by writing it into a temporary file. """
        with TemporaryFile() as f:
            pickle.dump(utils.DummyCatastrophe(), f)
