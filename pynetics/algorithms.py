from .catastrophe import Catastrophe
from .stop import StopCondition
from .utils import check_is_instance_of


class GeneticAlgorithm:
    """ Base class where the evolutionary algorithm works.

    More than one algorithm may exist so a base class is created for specify the
    contract required by the other classes to work properly.
    """

    def __init__(
            self,
            stop_condition,
            populations,
            catastrophe,
    ):
        """ Initializes the genetic algorithm with the defaults.

        The populations_desc param should follow certain rules in order to work
        in the way the genetic algorithm is intended:
        1.  The population size of each of the populations must be greater than
            or equal to 1. If not, there will be nothing to be evolved.
        2.  The replacement rate should be at least 1 (otherwise no individual
            will be replaced) and, at most, the population size (i.e. a total
            replacement also called generational scheme).
        3.  The spawning pool must be an instance of SpawningPool class (or any
            of their subclasses).
        4.  The fitness method must be an instance of FitnessMethod class (or
            any of its subclasses).

        :param stop_condition: The condition to be met in order to stop the
            genetic algorithm.
        :param populations: The populations to be evolved.
        :param catastrophe: The method to be used as catastrophe
            operation.
        :raises UnexpectedClassError: If any of the input variables doesn't
            follow the contract required (i.e. doesn't inherit from a predefined
            class).
        """
        self.__stop_condition = check_is_instance_of(
                stop_condition,
                StopCondition
        )
        self.__populations = populations
        self.__catastrophe = check_is_instance_of(catastrophe, Catastrophe)
        self.generation = 0

        for population in self.__populations:
            population.genetic_algorithm = self

    def run(self):
        """ Runs the simulation.

        The process is as follows: initialize populations and, while the stop
        condition is not met, do a new evolve step. This process relies in the
        abstract method "step".
        """
        self.__initialize()
        while not self.__stop_condition(self):
            for population in self.populations:
                population.evolve()
                self.__catastrophe(population)
            self.generation += 1

    def __initialize(self):
        """ Called when starting the genetic algorithm to initialize it. """
        self.__generation = 0

    @property
    def populations(self):
        """ Returns the populations being evolved. """
        return self.__populations