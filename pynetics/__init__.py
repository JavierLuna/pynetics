import collections
import collections.abc
import enum
import inspect
import math
import operator
import random
from abc import ABCMeta, abstractmethod
from typing import Callable, Iterable, Sequence, Any, Optional

from pynetics.utils import take_chances, clone_empty
from .exceptions import WrongValueForInterval, NotAProbabilityError, \
    PyneticsError, InvalidSizeError, RequiredValueError

__version__ = '0.4.1'


class Event(enum.Enum):
    """ What events may occur inside of any GeneticAlgorithm object. """
    ALGORITHM_START = enum.auto()
    ALGORITHM_END = enum.auto()
    STEP_START = enum.auto()
    STEP_END = enum.auto()


class GeneticAlgorithm(metaclass=ABCMeta):
    """ Base class with the definition of how a GA works.

    More than one algorithm may exist so a base class is created for specify the
    contract required by the other classes to work properly.
    """

    def __init__(self, stop_condition: Callable[['GeneticAlgorithm'], bool]):
        """ Initializes this object.

        It expects a criteria to be met in order to stop the algorithm.

        :param stop_condition: The criteria to be met after each of the steps
            performed by the algorithm to stop it. It could be any callable
            while complying the contract. The superclass
            :class:`stop.StopCondition` is compliant with the contract.
        """
        self.stop_condition = stop_condition
        self.listeners = collections.defaultdict(list)
        self.generation: int = 0

    def run(self):
        """ Runs the simulation.

        The process is as follows: initialize populations and, while the stop
        condition is not met, do a new evolve step. This process relies in the
        abstract method "step".
        """
        # Initializes the algorithm
        self.initialize()
        # Calls all the listeners righ before start the algorithm.
        [f(self) for f in self.listeners[Event.ALGORITHM_START]]
        # Start running the algorithm until the stop conditino is met.
        while not self.stop_condition(self):
            # Calls all the listeners registred before the step is performed.
            [f(self) for f in self.listeners[Event.STEP_START]]
            self.step()
            self.generation += 1
            # Calls all the listeners registred after the step is performed.
            [f(self) for f in self.listeners[Event.STEP_END]]
        # Calls all the listeners registred after the algorithm has ended.
        [f(self) for f in self.listeners[Event.ALGORITHM_END]]

    def initialize(self):
        """ Called when starting the genetic algorithm to initialize it. """
        self.generation = 0

    @abstractmethod
    def step(self):
        """ Performs an step in the genetic algorithm.

        Depending on the implementation, this step may be very different so
        there is no general implementation here.
        """

    @abstractmethod
    def best(self, generation: int = None) -> 'Individual':
        """ The best individual obtained.

        :param generation: The generation of the individual that we want to
            recover. If not set, this will be the one emerged in the last
            generation. Defaults to None (not set, thus last generation).
        :return: The best individual generated in the specified generation or in
            the last one if no generation is specified.
        """

    @abstractmethod
    def clone(self) -> 'GeneticAlgorithm':
        """ Creates an instance as an exact copy of this algorithm.

        The implementing subclass must override this method calling the super
        class method because it has some attributes also to be cloned.

        :return: An exact copy of this genetic algorithm.
        """
        ga = utils.clone_empty(self)
        ga.stop_condition = self.stop_condition
        ga.listeners = self.listeners
        ga.generation = self.generation
        return ga

    def on_start(self, f: Callable[['GeneticAlgorithm'], None]):
        """ Specifies a functor to be called when the algorithm starts.

        This function will be called AFTER initialization but BEFORE the first
        iteration, including the check against the stop condition.

        :param f: The functor to be called. It must accept a GeneticAlgorithm
            instance as a parameter.
        """
        ga = self.clone()
        ga.listeners[Event.ALGORITHM_START].append(f)
        return ga

    def on_end(self, f: Callable[['GeneticAlgorithm'], None]):
        """ Specifies a functor to be called when the algorithm ends.

        Particularly, this method will be called AFTER the stop condition
        has been met.

        :param f: The functor to be called. It must accept a GeneticAlgorithm
            instance as a parameter.
        """
        ga = self.clone()
        ga.listeners[Event.ALGORITHM_END].append(f)
        return ga

    def on_step_start(self, f: Callable[['GeneticAlgorithm'], None]):
        """ Specifies a functor to be called when an iteration step starts.

        This method will be called AFTER the stop condition has been checked
        and proved to be false) and BEFORE the new step is computed.

        :param f: The functor to be called. It must accept a GeneticAlgorithm
            instance as a parameter.
        """
        ga = self.clone()
        ga.listeners[Event.STEP_START].append(f)
        return ga

    def on_step_end(self, f: Callable[['GeneticAlgorithm'], None]):
        """ Specifies a functor to be called when an iteration ends.

        This method will be called AFTER an step of the algorithm has been
        computed and BEFORE a new check against the stop condition is going
        to be made.

        :param f: The functor to be called. It must accept a GeneticAlgorithm
            instance as a parameter.
        """
        ga = self.clone()
        ga.listeners[Event.STEP_END].append(f)
        return ga


class StopCondition(metaclass=ABCMeta):
    """ A condition to be met in order to stop the algorithm.

    Although the stop condition is defined as a class, it's enough to provide a
    function that is able to discern whether the time has come to stop (True or
    False) receiving as parameter the population.
    """

    @abstractmethod
    def __call__(self, ga: 'GeneticAlgorithm') -> bool:
        """ Checks if this stop condition is met.

        :param ga: The genetic algorithm where this stop condition belongs.
        :return: True if criteria is met, false otherwise.
        """


class Individual(metaclass=ABCMeta):
    """ One of the possible solutions to a problem.

    In a genetic algorithm, an individual is a tentative solution of a problem,
    i.e. the environment where populations of individuals evolve.
    """

    def __init__(self):
        """ Initializes the individual. """
        # TODO Implement an optional fitness cache
        self.population: Population = None
        self.f_method: Callable[['Individual'], float] = None

    def fitness(self) -> float:
        """ Computes the fitness of this individual.

        It will use the fitness method defined on its spawning pool.

        :return: A float value.
        """
        return self.f_method(self)

    def __str__(self):
        """ String representation of the individual.

        By default it'll be the string representation of the phenotype.
        """
        return str(self.phenotype())

    @abstractmethod
    def phenotype(self) -> Any:
        """ The expression of this particular individual in the environment.

        :return: An object representing this individual in the environment
        """

    @abstractmethod
    def clone(self) -> 'Individual':
        """ Creates an instance as an exact copy of this individual.

        The implementing subclass must override this method calling the super
        class method because it has some attributes also to be cloned.

        :return: A brand new individual like this one.
        """
        individual = clone_empty(self)
        individual.population = self.population
        individual.f_method = self.f_method
        return individual


class Diversity:
    """ Represents the diversity of a bunch of individuals. """

    @abstractmethod
    def __call__(self, individuals):
        """ It returns a value that symbolizes how diverse is the population.

        The expected value will rely completely over the Individual
        implementation.

        :param individuals: A sequence of individuals from which obtain the
            diversity.
        :return: A value representing the diversity.
        """


class SpawningPool(metaclass=ABCMeta):
    """ Defines the methods for creating individuals required by population. """

    def __init__(
            self,
            fitness: Callable[[Individual], float]
    ):
        """ Initializes the object.

        :param fitness: The fitness the individuals will have in order to be
            evaluated.
        """
        self.fitness = fitness

    def spawn(self) -> Individual:
        """ Returns a new random individual.

        It uses the abstract method "create" to be implemented with the logic
        of individual creation. The purpose of this method is to add the
        parameters the base individual needs.

        :return: An individual instance.
        """
        individual = self.create()
        individual.f_method = self.fitness
        return individual

    @abstractmethod
    def create(self) -> Individual:
        """ Creates a new individual randomly.

        :return: A new Individual object.
        """


class Population(collections.abc.MutableSequence):
    """ Manages a population of individuals.

    A population is where individuals of the same kind evolve over an
    environment. A basic genetic algorithm consists in a single population, but
    more complex schemes involve two or more populations evolving concurrently.
    """

    def __init__(
            self, *,
            size: int,
            spawning_pool: 'SpawningPool',
            diversity: 'Diversity',
            individuals: Optional[Iterable[Individual]] = None

    ):
        """ Initializes this population, optionally filling it with individuals.

        :param size: The numbers of individuals will have once initialized. A
            collection of $n$ individuals will be randomly generated in order to
            fulfill this requirement. This $n$ will be equals to the parameter
            size minus the number of individuals given in the parameter
            `individuals`. It should be greater than 0.
        :param individuals: Some starting individuals. If no individuals are
            provided or if its size is lower than the one specified by parameter
            `size`, some individuals will be randomly generated (see docs for
            parameter `size`. If it's greater, the only the first `size`
            individuals will be selected.
        :param spawning_pool: The object that generates individuals.
        :raises InvalidSizeError: If the provided size for the population is invalid.
        """
        if not size or size < 1:
            raise InvalidSizeError('> 0', size)
        if not spawning_pool:
            raise RequiredValueError('spawning_pool')
        if not diversity:
            raise RequiredValueError('diversity')

        self.initial_size = size
        self.individuals = []
        self.spawning_pool = spawning_pool
        self.diversity = diversity

        # Populate the individuals until `size`
        if individuals:
            for individual in individuals[:size]:
                self.append(individual)
        while len(self.individuals) < size:
            self.append(self.spawning_pool.spawn())

        self.__sorted = False

    def sort(self):
        """ Sorts this population from best to worst individual. """
        if not self.__sorted:
            self.individuals.sort(
                key=operator.methodcaller('fitness'),
                reverse=True
            )
            self.__sorted = True

    def __len__(self):
        """ Returns the number of individuals this population has. """
        return len(self.individuals)

    def __delitem__(self, i):
        """ Removes the $i$-th individual from the population.

        The population will be sorted by its fitness before deleting.

        :param i: The ith individual to delete.
        """
        del self.individuals[i]

    def __setitem__(self, i, individual):
        """ Puts the named individual in the ith position.

        This call will cause a new sorting of the individuals the next time an
        access is required. This means that is preferable to make all the
        inserts in the population at once instead doing interleaved readings and
        inserts.

        :param i: The position where to insert the individual.
        :param individual: The individual to be inserted.
        """
        individual.population = self
        self.individuals.__setitem__(i, individual)

        self.__sorted = False

    def insert(self, i, individual):
        """ Ads a new element to the ith position of the population population.

        This call will cause a new sorting of the individuals the next time an
        access is required. This means that is preferable to make all the
        inserts in the population at once instead doing interleaved readings and
        inserts.

        :param i: The position where insert the individual.
        :param individual: The individual to be inserted in the population
        """
        individual.population = self
        self.individuals.insert(i, individual)

        self.__sorted = False

    def __getitem__(self, i):
        """ Returns the individual located on the ith position.

        The population will be sorted before accessing to the element so it's
        correct to assume that the individuals are arranged from fittest (i = 0)
        to least fit (n  len(populaton)).

        :param i: The index of the individual to retrieve.
        :return: The individual.
        """
        return self.individuals.__getitem__(i)

    def best(self):
        """ Returns the best individual for the gth.

        :return: The best individual for that generation.
        """
        self.sort()
        return self[0]


class Evolver:
    """ Objects of this class performs evolution steps over populations. """

    def __init__(
            self, *,
            selection: 'Selection',
            recombination: 'Recombination',
            p_recombination: float,
            mutation: 'Mutation',
            p_mutation: float,
            replacement: 'Replacement',
            replacement_rate: float
    ):
        """ Initializes this instance.

        :param selection: The method which perform the selection of individuals.
        :param recombination: The method to recombine parents in order to
            generate an offspring with characteristics of the parents.
        :param p_recombination: The odds for recombination method to be
            performed over a set of selected individuals to generate progeny. If
            not performed, progeny will be the parents. Must be a value between
            0 and 1 (both included). If not provided, defaults to 1.0.
        :param mutation: The method to mutate an individual.
        :param p_mutation: The odds for mutation method to be performed over a
            progeny. It's applied once for each individual. If not performed the
            individuals will not be modified. Must be a value between 0 and 1
            (both included). If not provided, it defaults to 0.0 (no mutation is
            performed).
        :param replacement: The method that will replace the individuals of the
            population with the new offspring.
        :param replacement_rate: The rate of individuals to be replaced in each
            step of the algorithm. Must be a float value in the (0, 1] interval.
        :raises WrongValueForIntervalError: If any of the bounded values fall
            out of their respective intervals.
        :raises NotAProbabilityError: If a value was expected to be a
            probability and it wasn't.
        :raises UnexpectedClassError: If any of the input variables doesn't
            follow the contract required (i.e. doesn't inherit from a predefined
            class).
        """
        self.selection = selection
        self.recombination = recombination
        self.mutation = mutation or NoMutation()
        self.replacement = replacement
        self.replacement_rate = replacement_rate
        self.p_recombination = p_recombination
        self.p_mutation = p_mutation

        self.selection_size = len(
            inspect.signature(recombination.__call__).parameters
        )

    def __call__(
            self, *,
            population: 'Population',
            steps: int = 1
    ):
        """ Performs evolution steps over a population.

        :param population: The population on which apply the evolution steps.
        :param steps: The number of evolution steps to execute over the
            population. It defaults to one.
        """
        if steps < 1:
            raise InvalidSizeError('> 0', steps)
        # The offspring size is determined by the length of the population and
        # the replacement rate.
        offspring_size = int(math.ceil(len(population) * self.replacement_rate))
        offspring = []
        for step in range(steps):
            offspring.clear()
            while len(offspring) < offspring_size:
                # Selection
                parents = self.selection(population, self.selection_size)
                # Recombination
                if take_chances(self.p_recombination):
                    progeny = self.recombination(*parents)
                else:
                    progeny = [i.clone() for i in parents]
                # Mutation
                individuals_who_fit = min(
                    len(progeny),
                    offspring_size - len(offspring)
                )
                progeny = [
                    self.mutation(individual, self.p_mutation)
                    for individual in
                    random.sample(progeny, individuals_who_fit)
                    ]
                # Add progeny to the offspring
                offspring.extend(progeny)

            # Once offspring is generated, a replace step is performed
            self.replacement(population, offspring)


class Fitness(metaclass=ABCMeta):
    """ Estimates how adapted is the individual to the environment. """

    @abstractmethod
    def __call__(self, individual: 'Individual') -> float:
        """ Estimates how adapted is the individual.

        :param individual: The individual to which estimate the adaptation.
        :return: A float value pointing out the adaption to the environment. The
            higher the value, the more adapted is the individual to the
            environment.
        """


class Selection(metaclass=ABCMeta):
    """ Selection of the fittest individuals among the population.

    The selection method is defined as a class. However, it is enough to provide
    as a selection method, i.e. a function that receives a sequence and a number
    of individuals, and returns a sample of individuals of that size from the
    given population.
    """

    def __call__(
            self, *,
            population: Population,
            n: int = 1
    ) -> Sequence[Individual]:
        """ Makes some checks to the configuration before delegating selection.

        After checking the parameters, the selection is performed by perform
        method.

        :param population: The sequence of individuals from which select the
            individuals.
        :param n: The number of individuals to return.
        :return: A sequence of individuals.
        :raises InvalidSizeError: If length of the population is smaller than the
            number of individuals to select and the repetition parameter is set
            to False (i.e. the same Individual cannot be selected twice or more
            times).
        """
        num_individuals = len(population)
        if not 0 < n <= num_individuals:
            raise InvalidSizeError(f'0 < n <= {num_individuals}', n)
        else:
            return self.perform(population=population, n=n)

    @abstractmethod
    def perform(
            self, *,
            population: Population,
            n: int
    ) -> Sequence[Individual]:
        """ It makes the selection according to the subclass implementation.

        :param population: A sequence of individuals from which select some.
        :param n: The number of individuals to return.
        :return: A sequence of $n$ individuals.
        """


class Recombination(metaclass=ABCMeta):
    """ Defines the behaviour of a recombination operator.

    A recombination operator takes a set of individuals (i.e. parents) and
    generates a different set of individuals (i.e. offspring) normally with
    aspects derived from their parents.
    """

    @abstractmethod
    def __call__(
            self,
            *individuals: Iterable['Individual']
    ) -> Sequence['Individual']:
        """ Implementation of the recombine method.

        :param individuals: One or more Individual instances to use as parents
            in the recombination.
        :return: A sequence of individuals with characteristics of the parents.
        """


class Mutation(metaclass=ABCMeta):
    """ Behaviour of a genetic algorithm mutation operator. """

    @abstractmethod
    def __call__(self, individual: 'Individual', p: float) -> Individual:
        """ Applies the mutation method to the individual.

        :param individual: an individual to mutate.
        :param p: The probability of mutation.
        :return: A cloned individual of the one passed as parameter but with a
            slightly (or not, X-MEN!!!!) mutation.
        """


class NoMutation(Mutation):
    """ An utility function where no mutation takes place."""

    def __call__(self, individual, p):
        """ No mutation is performed. Instead, an exact clone is returned.

        :param individual: an individual to mutate.
        :param p: The probability of mutation.
        :return: A cloned individual of the one passed as parameter but with a
            slightly (or not, X-MEN!!!!) mutation.
        """
        return individual.clone()


class Replacement(metaclass=ABCMeta):
    """ Replacement of individuals of the population. """

    def __call__(
            self, *,
            population: Population,
            offspring: Sequence[Individual]
    ) -> None:
        """ Realizes the parameter checkings and performs the replacement.

        This method relies in the `perform` method.

        :param population: The population where make the replacement.
        :param offspring: The new individuals to repaces the old ones. If no
            individuals are provided, no replacement take place and the
            population will remain untouched.
        :raise RequiredValueError: If no population is provided.
        """
        if not population:
            raise RequiredValueError('population')

        if offspring:
            self.perform(population=population, offspring=offspring)

    @abstractmethod
    def perform(
            self, *,
            population: 'Population',
            offspring: Sequence['Individual']
    ) -> None:
        """ Performs the replacement, according to ths subclass implementation.

        This method will be called from __call__ after some parameter checking
        so is preferable to override this instead of __call__.

        :param population: The original population where to replace individuals.
        :param offspring: The individuals provided to replace the old ones in
            the population. It is guaranteed that there will be at least one
            individuals in the sequence.
        """


class Catastrophe(metaclass=ABCMeta):
    """ Behaviour of a genetic algorithm catastrophe operator.

    It's expected for this operator to keep track of the GeneticAlgorithm state
    and know when to act since it will be called every step of the algorithm
    after replacement operation.
    """

    @abstractmethod
    def __call__(self, population: 'Population'):
        """ Applies the catastrophe to the specified population.

        :param population: The population where to apply the catastrophic method.
        """
