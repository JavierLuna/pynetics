import operator
import random
from typing import Sequence

from . import Selection, Population, Individual


class BestIndividual(Selection):
    """ Selects the best individuals among the population. """

    def perform(
            self, *,
            population: Population,
            n: int
    ) -> Sequence[Individual]:
        """ Gets the top n individuals out of all the population.

        :param population: The population from which select the individuals.
        :param n: The number of individuals to return.
        :return: A list of n individuals.
        """
        return population[:n]


class ProportionalToFitness(Selection):
    """ Selects individuals randomly proportionally to their fitness. """

    def perform(
            self, *,
            population: Population,
            n: int
    ) -> Sequence[Individual]:
        """ Gets randomly the population, giving more probability based on the
        their probability. The higher the probability, the higher the chance to
        be selected.

        :param n: The number of population to return.
        :param population: The population from which select the population.
        :return: A list of n population.
        """
        # TODO Implement
        raise NotImplementedError()


class ProportionalToPosition(Selection):
    """ Selects individuals randomly proportionally to their positions. """

    def perform(
            self, *,
            population: Population,
            n: int
    ) -> Sequence[Individual]:
        """ Gets randomly the individuals, giving more probability to those in
        first positions of the population, i.e. those fittest.

        The probability to be selected is proportional to the position of the
        fitness of the individual among the population (i.e. those with better
        fitness have better positions, but a very high fitness doesn't implies
        more chances to be selected).

        :param n: The number of individuals to return.
        :param population: The population from which select the individuals.
        :return: A list of n individuals.
        """
        # TODO Implement
        raise NotImplementedError()


class Tournament(Selection):
    """ Selects best individuals of a random sample of the whole population. """

    def __init__(self, sample_size):
        """ Initializes this selector.

        :param sample_size: The size of the random sample of individuals to pick
            prior to make the selection of the fittest.
        """
        self.sample_size = sample_size

    def perform(
            self, *,
            population: Population,
            n: int
    ) -> Sequence[Individual]:
        """ Gets the best individuals from a random sample of the population.

        To do it, a sample of individuals will be selected randomly and, after
        that, the best individual of the sample is then selected. This process
        (i.e. extract sample and the get best individual from sample) is done
        as many times as individuals to be selected.

        :param n: The number of individuals to return.
        :param population: The population from which select the individuals.
        :return: A list of n individuals.
        """
        individuals = []
        for _ in range(n):
            sample = random.sample(population, self.sample_size)
            individuals.append(
                max(sample, key=operator.methodcaller('fitness'))
            )
        return individuals


class Uniform(Selection):
    """ Selects individuals randomly from the population. """

    def perform(
            self, *,
            population: Population,
            n: int
    ) -> Sequence[Individual]:
        """ Selects n individuals randomly from the population.

        The selection is done by following a uniform distribution along the
        entire population.

        :param population: The population from which select the individuals.
        :param n: The number of individuals to return.
        :return: A list of n individuals.
        """
        return random.sample(population, n)
