from unittest import TestCase

from pynetics import Population, SpawningPool, FitnessMethod
from pynetics.ga_list import ListIndividual
from pynetics.selection import BestIndividual


class DummySpawningPool(SpawningPool):
    """ A spawning pool just for tests. """

    def __init__(self, individuals):
        """ Initializes this spawning pool with 10 predefined individuals. """
        self.i = 0
        self.n = len(individuals)
        self.individuals = individuals[:]

    def create(self):
        """ Return one of the individuals in the list, consecutively. """
        if self.i == self.n:
            self.i = 0
        individual = self.individuals[self.i % self.n]
        self.i += 1
        return individual


class DummyFitness(FitnessMethod):
    """ A fitness just for tests. """

    def perform(self, individual):
        """ The more 1 an individual has in its list, the fittest it is. """
        return sum(x for x in individual)


class BestIndividualTestCase(TestCase):
    """ Tests for BestIndividual selection method. """

    def individuals(self, n):
        individuals = []
        for i in range(10):
            individual = ListIndividual()
            for j in range(n):
                individual.append(0 if j <= i else 1)
            individuals.append(individual)
        return individuals

    def test_when_population_size_is_lower_than_selection_size(self):
        """ Cannot select more individuals than population size. """
        population_size = 10
        population = Population(
            population_size,
            5,
            DummySpawningPool(self.individuals(population_size)),
            DummyFitness(),
        )
        population.initialize()
        with self.assertRaises(ValueError):
            BestIndividual()(population, population_size * 2)

    def test_when_population_size_is_lower_than_selection_size_with_rep(self):
        """ Best individual is selected as many times as selection size. """
        population_size = 10
        selection_size = population_size * 2
        population = Population(
            population_size,
            5,
            DummySpawningPool(self.individuals(population_size)),
            DummyFitness(),
        )
        population.initialize()
        individuals = BestIndividual(rep=True)(population, selection_size)
        self.assertEquals(len(individuals), selection_size)
        for individual in individuals:
            self.assertEquals(individual, population[0])

    def test_when_population_size_is_equals_to_selection_size(self):
        """ All the population is returned. """
        population_size = 10
        individuals = self.individuals(population_size)
        population = Population(
            population_size,
            5,
            DummySpawningPool(individuals),
            DummyFitness(),
        )
        population.initialize()
        selected_individuals = BestIndividual()(population, population_size)
        for individual in individuals:
            self.assertIn(individual, selected_individuals)
        for individual in selected_individuals:
            self.assertIn(individual, individuals)

    def test_when_population_size_is_equals_to_selection_size_with_rep(self):
        """ Best individual is returned as many times as population size. """
        population_size = 10
        selection_size = 10
        individuals = self.individuals(population_size)
        population = Population(
            population_size,
            5,
            DummySpawningPool(individuals),
            DummyFitness(),
        )
        population.initialize()
        selected_individuals = BestIndividual(rep=True)(
            population,
            selection_size
        )
        self.assertEquals(len(individuals), selection_size)
        self.assertEquals(len(individuals), len(selected_individuals))
        for individual in selected_individuals:
            self.assertEquals(individual, population[0])

    def test_when_population_size_is_bigger_than_selection_size(self):
        """ The best individuals are returned. """
        population_size = 10
        selection_size = int(population_size / 2)
        population = Population(
            population_size,
            5,
            DummySpawningPool(self.individuals(population_size)),
            DummyFitness(),
        )
        population.initialize()
        individuals = BestIndividual()(population, selection_size)
        self.assertEquals(len(individuals), selection_size)
        for i in range(len(individuals)):
            self.assertEquals(individuals[i], population[i])

    def test_when_population_size_is_bigger_than_selection_size_with_rep(self):
        """ Best individual is returned as many times as selection size. """
        population_size = 10
        selection_size = int(population_size / 2)
        population = Population(
            population_size,
            5,
            DummySpawningPool(self.individuals(population_size)),
            DummyFitness(),
        )
        population.initialize()
        individuals = BestIndividual(rep=True)(population, selection_size)
        self.assertEquals(len(individuals), selection_size)
        for individual in individuals:
            self.assertEquals(individual, population[0])
