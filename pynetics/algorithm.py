import abc
import inspect
import random
from typing import Callable, Tuple

from pynetics.utils import take_chances


class SpawningPool(metaclass=abc.ABCMeta):
    """ Defines the methods for creating individuals required by population. """

    @abc.abstractmethod
    def create(self) -> Individual:
        """ Creates a new individual randomly.

        :return: A new Individual object.
        """


class Population(list):
    """ Manages a population of individuals. """

    def __init__(
            self,
            spawning_pool: SpawningPool,
            size: int,
            *args,
            **kwargs
    ):
        """ Initializes the population, filling it with individuals

        Because operators requires to know which individual is the fittest, others which is the less fit and others need
        to travel along the collection of individuals in some way or another (e.g. from fittest to less fit), the
        population is always sorted when an access is required. Thus, writing population[0] always returns the fittest
        individual, population[1] the next and so on, until population[-1] which is the less fit.

        :param spawning_pool: The object that generates individuals.
        :param size: The size this population should have.
        """
        super().__init__(*args, **kwargs)
        self.__size = size
        self.__spawning_pool = spawning_pool
        self.__sorted = True
        [self.append(i) for i in spawning_pool.create()]

    def sort(self, *args, **kwargs):
        """ Sorts the list of individuals by its fitness. """
        if not self.__sorted:
            super().sort(*args, **kwargs)
            self.__sorted = True

    def __getitem__(self, index: int) -> Individual:
        """ Returns the individual located on this position.

        Treat this call as if population were sorted by fitness, from the fittest to the less fit.

        :param index: The index of the individual to recover.
        :return: The individual.
        """
        self.sort()
        return super().__getitem__(index)

    def __setitem__(self, index: int, individual: Individual):
        """ Puts the named individual in the specified position.

        This call will cause a new sorting of the individuals the next time an access is required. This means that is
        preferable to make all the inserts in the population at once instead doing interleaved readings and inserts.

        :param index: The position where to insert the individual.
        :param individual: The individual to be inserted.
        """
        self.__sorted = False
        self.__setitem__(index, individual)

    def extend(self, individuals: Tuple[Individual, ...]):
        """ Extends the population with a collection of individuals.

        This call will cause a new sorting of the individuals the next time an access is required. This means that is
        preferable to make all the inserts in the population at once instead doing interleaved readings and inserts.

        :param individuals: A collection of individuals to be inserted into the population.
        """
        self.__sorted = False
        self.extend(individuals)


class StopCriteria(metaclass=abc.ABCMeta):
    """ A criteria to be met in order to stop the algorithm. """

    @abc.abstractmethod
    def __call__(self, population: Population):
        """ Checks if this stop criteria is met.

        :param population: The genetic algorithm to check.
        :return: True if criteria is met, false otherwise.
        """


class SelectMethod(metaclass=abc.ABCMeta):
    """ Selection of the fittest individuals among the population. """

    @abc.abstractmethod
    def __call__(self, population: Population, n: int) -> Tuple[Individual, ...]:
        """ It makes the selection according to the subclass implementation.

        :param n: The number of individuals to return.
        :param population: The population from which select the individuals.
        :return: A list of individuals.
        """


class ReplaceMethod(metaclass=abc.ABCMeta):
    """ Replacement of individuals of the population. """

    @abc.abstractmethod
    def __call__(self, population: Population, offspring: Population) -> None:
        """ It makes the replacement according to the subclass implementation.

        :param population: The population where make the replacement.
        :param offspring: The new population to use as replacement.
        """


class CrossoverMethod(metaclass=abc.ABCMeta):
    """ Defines the behaviour of a genetic algorithm crossover operator. """

    @abc.abstractmethod
    def __call__(self, individuals: Tuple[Individual, ...]) -> Tuple[Individual, ...]:
        """ Implementation of the crossover operation.

        The crossover implementation must be aware of the individual type. Given that not all the implementations are
        the same, not all the crossover operations may work.

        :param individuals: A sequence of individuals to cross.
        :returns: A sequence of individuals with characteristics of the parents.
        """


class MutateMethod(metaclass=abc.ABCMeta):
    """ Defines the behaviour of a genetic algorithm mutation operator. """

    @abc.abstractmethod
    def __call__(self, individual: Individual) -> None:
        """ Implementation of the mutation operation.

        The mutation implementation must be aware of the implementation type. Given that not all the implementations are
        the same, not all the mutation operations may work.

        :param individual: an individual to mutate.
        :returns: A new mutated individual.
        """


class CatastropheMethod(metaclass=abc.ABCMeta):
    """ Defines the behaviour of a genetic algorithm catastrophe operator.

    It's expected that this operator keep track of when to act, since it will be called every step of the algorithm
    after replacement operation.
    """

    @abc.abstractmethod
    def __call__(self, population: Population) -> None:
        """ Implementation of the catastrophe operation.

        :param population: the population which may suffer the catastrophe
        """


class Individual(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fitness(self):
        """ It indicates how the individual is adapted to the environment.

        :return: Depends on the problem, it's not tied to any type.
        """

    @abc.abstractmethod
    def phenotype(self):
        """ The representation of this individual as a solution of our problem.

        :return: Depends on the problem, it's not tied to any representation.
        """

    @abc.abstractmethod
    def __lt__(self, other: 'Individual') -> bool:
        """ Compares two individuals checking if the former is lower than the latter.

        :param other: The other individual to compare to.
        :return: A value of True if current individual is worst than the specified by parameter and False otherwise.
        """

    @abc.abstractmethod
    def __eq__(self, other: 'Individual') -> bool:
        """ Compares two individuals checking if they're or not equal.

        :param other: The other individual to compare to.
        :return: A value of True if they're equal and False otherwise.
        """


class GenericGA:
    """ A generic genetic algorithm.

    Different algorithms can be implemented just by changing the initialization parameters (e.g. steady-state algorithms
    vs generational changing the value of replacement_rate.
    """

    def __init__(
            self,
            population_size: int,
            replacement_rate: int,
            spawning_pool: SpawningPool,
            stop_criteria: Callable[[Population], bool],
            select_method: Callable[[Population, int], Tuple[Individual, ...]],
            replace_method: Callable[[Population, Population], None],
            crossover_method: Callable[[Tuple[Individual, ...]], Tuple[Individual, ...]],
            mutate_method: Callable[[Individual], None],
            catastrophe_method: Callable[[Population], None],
            p_crossover: float,
            p_mutation: float,
    ):
        """ Initializes this particular algorithm

        :param population_size: The number of individuals that the population evolved in this algorithm will have.
        :param replacement_rate: The number of individuals to be replaced in each step of the algorithm. It should be
            lower than the population type.
        :param spawning_pool: The functor that will create the individuals when needed.
        :param stop_criteria: The criteria to be met in order to stop the genetic algorithm.
        :param select_method: The method to be used as selection scheme.
        :param replace_method: The method to be used as replacement scheme.
        :param crossover_method: The method to be used as crossover operator scheme.
        :param mutate_method: The method to be used as mutation operator scheme.
        :param catastrophe_method: The method to be used as catastrophe operation.
        :param p_crossover: Probability that individuals, after being selected, cross each other to produce offspring.
        :param p_mutation: Probability that an individual mutates.
        :raises ValueError: Just in case an input value is not valid. See the description of the input values for more
            information.
        """
        # TODO Validate input data
        self.__p_size = population_size
        self.__replacement_rate = replacement_rate
        self.__spawning_pool = spawning_pool
        self.__stop = stop_criteria
        self.__select = select_method
        self.__replace = replace_method
        self.__cross = crossover_method
        self.__mutate = mutate_method
        self.__catastrophe = catastrophe_method
        self.__p_crossover = p_crossover
        self.__p_mutation = p_mutation

        self.population = None

    def run(self):
        self.population = Population(self.__spawning_pool, self.__p_size)
        while not self.__stop(self.population):
            offspring = Population(self.__spawning_pool, 0)
            while len(offspring) < self.__replacement_rate:
                # Selection
                individuals = self.__select(self.population, len(inspect.signature(self.__cross).parameters))
                # Crossover
                progeny = self.__cross(individuals) if take_chances(self.__p_crossover) else individuals
                # We take only the children needed to fulfill the new population
                progeny = random.sample(progeny, min(len(progeny), self.__replacement_rate - len(offspring)))
                # Mutation
                [self.__mutate(individual) for individual in progeny if take_chances(self.__p_mutation)]
                # We add the generated children to the offspring
                offspring.extend(progeny)
            # Replacement
            self.__replace(self.population, offspring)
            # Catastrophe
            self.__catastrophe(self.population)
