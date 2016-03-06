import abc

# TODO I'm not proud of this methods. I think they performance may be improved.
from .utils import take_chances


class ProbabilityBasedCatastrophe(Catastrophe, metaclass=abc.ABCMeta):
    """ Base class for some bundled probability based catastrophe methods.

    This method will have a probability to be triggered. Is expected this
    probability to be very little.
    """

    def __init__(self, probability):
        """ Initializes this catastrophe method.

        :param probability: The probability fot the catastrophe to happen.
        """
        self.__probability = probability

    def perform(self, population):
        if take_chances(self.__probability):
            self.perform_catastrophe()

    @abc.abstractmethod
    def perform_catastrophe(self, population):
        """ Returns a list of the individuals to remove from population.

        :param population: The population from where extract individuals.
        :return: The individuals to retain after the catastrophe application.
        """


class PackingByProbability(ProbabilityBasedCatastrophe):
    """ Replaces all repeated individuals maintaining only one copy of each. """

    def perform_catastrophe(self, population):
        """ Replaces all repeated individuals by new ones.

        :param population: The population where apply the catastrophe.
        """
        visited_individuals = []
        for i in range(len(population)):
            if population[i] in visited_individuals:
                population[i] = population.spawning_pool.spawn()
            visited_individuals.append(population[i])


class DoomsdayByProbability(ProbabilityBasedCatastrophe):
    """ Replaces all but the best individual. """

    def perform_catastrophe(self, population):
        """ Replaces all the individuals but the best.

        :param population: The population where apply the catastrophe.
        """
        for i in range(1, len(population)):
            population[i] = population.spawning_pool.spawn()
