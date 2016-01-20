import abc


class Individual:
    """ One of the possible solutions to a problem.

    In a genetic algorithm, an individual is a tentative solution of a problem,
    i.e. the environment where populations of individuals evolve.
    """

    def __init__(self):
        """ Initializes the individual. """
        self.population = None


class SpawningPool(metaclass=abc.ABCMeta):
    """ Defines the methods for creating individuals required by population. """

    @abc.abstractmethod
    def create(self):
        """ Creates a new individual randomly.

        :return: A new Individual object.
        """
