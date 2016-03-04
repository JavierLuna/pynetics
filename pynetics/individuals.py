import abc

from pynetics.fitnesses import Fitness


class Individual(metaclass=abc.ABCMeta):
    """ One of the possible solutions to a problem.

    In a genetic algorithm, an individual is a tentative solution of a problem,
    i.e. the environment where populations of individuals evolve.
    """

    # TODO Cache phenotype
    # TODO ¿Maybe a clone method?

    def __init__(self, disable_cache: bool = False):
        """ Initializes the individual.

        An individual contains a cache for the fitness method that prevents to
        compute it over and over again. However, as well as it is possible to
        clear this cache, also it is possible to disable it.

        :param disable_cache: Disables the fitness cache. Defaults to True,
            which means the cache is enabled.
        """
        self.disable_cache = disable_cache
        self.population = None
        self.fitness_method = None
        self.fitness_cached = None

    def fitness(self, init: bool = False) -> float:
        """ Computes the fitness of this individual.

        It will use the fitness method defined on its spawning pool.

        :param init: If this call to fitness is in initialization time. It
            defaults to False.
        :return: A float value.
        """
        if self.disable_cache or (not self.fitness_cached and init):
            return self.fitness_method(self, init)
        elif not self.fitness_cached:
            self.fitness_cached = self.fitness_method(self, init)
        return self.fitness_cached

    @abc.abstractmethod
    def phenotype(self):
        """ The expression of this particular individual in the environment.

        :return: An object representing this individual in the environment
        """


class SpawningPool(metaclass=abc.ABCMeta):
    """ Defines the methods for creating individuals required by population. """

    def __init__(self, fitness: Fitness):
        """ Initializes this spawning pool.

        :param fitness: The method to evaluate individuals.
        """
        self.population = None
        self.fitness = fitness

    def spawn(self) -> Individual:
        """ Returns a new random individual.

        It uses the abstract method "create" to be implemented with the logic
        of individual creation. The purpose of this method is to add the
        parameters the base individual needs.

        :return: An individual instance.
        """
        individual = self.create()
        individual.population = self.population
        individual.f_fitness = self.fitness
        return individual

    @abc.abstractmethod
    def create(self) -> Individual:
        """ Creates a new individual randomly.

        :return: A new Individual object.
        """
