import inspect
import math
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterable, Callable, Sequence

import operator
import random
from abc import ABCMeta, abstractmethod
from collections import abc

from pynetics.utils import take_chances, clone_empty
from .exceptions import WrongValueForInterval, NotAProbabilityError, \
    PyneticsError, InvalidSize

__version__ = '0.3.4'


class GeneticAlgorithm(metaclass=ABCMeta):
    """ Base class with the definition of how a GA works.

    More than one algorithm may exist so a base class is created for specify the
    contract required by the other classes to work properly.
    """

    def __init__(
        self,
        stop_condition: Callable[['GeneticAlgorithm'], bool]
    ):
        self.stop_condition = stop_condition
        self.generation = 0
        self.listeners = []

    @abstractmethod
    def run(self):
        """ Runs the simulation.

        The process is as follows: initialize populations and, while the stop
        condition is not met, do a new evolve step. This process relies in the
        abstract method "step".
        """
        self.initialize()
        [l.algorithm_started(self) for l in self.listeners]
        while not self.stop_condition(self):
            [l.step_started(self) for l in self.listeners]
            self.step()
            self.generation += 1
            [l.step_finished(self) for l in self.listeners]
        self.finish()
        [l.algorithm_finished(self) for l in self.listeners]

    def initialize(self):
        """ Called when starting the genetic algorithm to initialize it. """
        self.generation = 0

    def step(self):
        """ Called on every iteration of the algorithm. """
        pass

    def finish(self):
        """ Called one the algorithm has finished. """
        pass

    @abstractmethod
    def best(self) -> 'Individual':
        """ Returns the best individual obtained until this moment.

        :return: The best individual generated.
        """


class GAListener:
    """ A class that reacts to the events caused by the genetic algorithm. """

    def algorithm_started(self, ga: GeneticAlgorithm):
        """ Called when the algorithm start.

        This method will be called AFTER initialization but BEFORE the first
        iteration, including the check against the stop condition.

        :param ga: The GeneticAlgorithm instanced that called this method.
        """
        pass

    def algorithm_finished(self, ga: GeneticAlgorithm):
        """ Called when the algorithm finishes.

        Particularly, this method will be called AFTER the stop condition has
        been met.

        :param ga: The GeneticAlgorithm instanced that called this method.
        """
        pass

    def step_started(self, ga: GeneticAlgorithm):
        """ Called when a new step of the genetic algorithm starts.

        This method will be called AFTER the stop condition has been checked
        and proved to be false) and BEFORE the new step is computed.

        :param ga: The GeneticAlgorithm instanced that called this method.
        """
        pass

    def step_finished(self, ga: GeneticAlgorithm):
        """ Called when a new step of the genetic algorithm finishes.

        This method will be called AFTER an step of the algorithm has been
        computed and BEFORE a new check against the stop condition is going to
        be made.

        :param ga: The GeneticAlgorithm instanced that called this method.
        """
        pass


class StopCondition(metaclass=ABCMeta):
    """ A condition to be met in order to stop the algorithm.

    Although the stop condition is defined as a class, it's enough to provide a
    function that is able to discern whether the time has come to stop (True or
    False) receiving as parameter the population.
    """

    @abstractmethod
    def __call__(self, genetic_algorithm: GeneticAlgorithm) -> bool:
        """ Checks if this stop condition is met.

        :param genetic_algorithm: The genetic algorithm where this stop
            condition belongs.
        :return: True if criteria is met, false otherwise.
        """


class Individual(metaclass=ABCMeta):
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
        self.cache_disabled = disable_cache
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
        if self.cache_disabled or (not self.fitness_cached and init):
            return self.fitness_method(self, init)
        elif not self.fitness_cached:
            self.fitness_cached = self.fitness_method(self, init)
        return self.fitness_cached

    @abstractmethod
    def phenotype(self) -> Any:
        """ The expression of this particular individual in the environment.

        :return: An object representing this individual in the environment
        """

    def clone(self):
        """ Creates an instance as an exact copy of this individual

        If the implementing subclass has internal attributes to be cloned, the
        attributes copy should be implemented in an overriden version of this
        method.

        :return: A brand new individual like this one.
        """
        individual = clone_empty(self)
        individual.cache_disabled = self.cache_disabled
        individual.population = self.population
        individual.fitness_method = None
        individual.fitness_cached = None
        return individual


class SpawningPool(metaclass=ABCMeta):
    """ Defines the methods for creating individuals required by population. """

    def __init__(self, fitness: 'Fitness'):
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
        individual.fitness_method = self.fitness
        return individual

    @abstractmethod
    def create(self) -> Individual:
        """ Creates a new individual randomly.

        :return: A new Individual object.
        """


class Population(abc.MutableSequence):
    # TODO What if population where splitten in two (normal and sorted).
    """ Manages a population of individuals.

    A population is where individuals of the same kind evolve over an
    environment. A basic genetic algorithm consists in a single population, but
    more complex schemes involve two or more populations evolving concurrently.
    """

    def __init__(
        self,
        name: str = None,
        size: int = None,
        replacement_rate: float = None,
        spawning_pool: SpawningPool = None,
        selection: 'Selection' = None,
        recombination: 'Recombination' = None,
        p_recombination: float = None,
        mutation: 'Mutation' = None,
        p_mutation: float = None,
        replacement: 'Replacement' = None,
        individuals: Iterable[Individual] = None,
    ):
        """ Initializes the population, filling it with individuals.

        When the population is initialized, the fitness of the individuals
        generated is also calculated, implying that init_perform of every
        individual is called.

        Because operators requires to know which individual is the fittest,
        others which is the less fit and others need to travel along the
        collection of individuals in some way or another (e.g. from fittest to
        less fit), the population is always sorted when an access is required.
        Thus, writing population[0] always returns the fittest individual,
        population[1] the next and so on, until population[-1] which is the less
        fit.

        :param name: The name of this population.
        :param size: The size this population should have.
        :param replacement_rate: The rate of individuals to be replaced in each
            step of the algorithm. Must be a float value in the (0, 1] interval.
        :param spawning_pool: The object that generates individuals.
        :param selection: The method to select individuals of the population to
            recombine.
        :param recombination: The method to recombine parents in order to
            generate an offspring with characteristics of the parents. If none,
            no recombination will be applied.
        :param p_recombination: The odds for recombination method to be
            performed over a set of selected individuals to generate progeny. If
            not performed, progeny will be the parents. Must be a value between
            0 and 1 (both included).
        :param mutation: The method to mutate an individual. If none, no
            mutation over the individual will be applied.
        :param p_mutation: The odds for mutation method to be performed over a
            progeny. It's applied once for each individual. If not performed the
            individuals will not be modified. Must be a value between 0 and 1
            (both included).
        :param replacement: The method that will add and remove individuals from
            the population given the set of old individuals (i.e. the ones on
            the population before the evolution step) and new individuals (i.e.
            the offspring).
        :param individuals: The list of starting individuals. If none or if its
            length is lower than the population size, the rest of individuals
            will be generated randomly. If the length of initial individuals is
            greater than the population size, a random sample of the individuals
            is selected as members of population.
        :raises ValueError: If no name for this population is provided.
        :raises WrongValueForIntervalError: If any of the bounded values fall
            out of their respective intervals.
        :raises NotAProbabilityError: If a value was expected to be a
            probability and it wasn't.
        :raises UnexpectedClassError: If any of the instances provided wasn't
            of the required class.
        """
        if not name:
            raise PyneticsError('A name for population is required')
        if size is None or size < 1:
            raise InvalidSize('> 0', size)
        if replacement_rate is None or not 0 < replacement_rate <= 1:
            raise WrongValueForInterval(
                'replacement_rate',
                0,
                1,
                replacement_rate,
                inc_lower=False
            )
        if p_recombination is None or not 0 <= p_recombination <= 1:
            raise NotAProbabilityError('p_recombination', p_recombination)
        if p_mutation is None or not 0 <= p_mutation <= 1:
            raise NotAProbabilityError('p_mutation', p_mutation)

        self.name = name
        self.size = size
        self.replacement_rate = replacement_rate
        self.spawning_pool = spawning_pool
        self.spawning_pool.population = self
        self.selection = selection
        self.recombination = recombination
        self.p_recombination = p_recombination
        self.mutation = mutation
        self.p_mutation = p_mutation
        self.replacement = replacement

        if individuals is not None:
            self.individuals = [i.clone() for i in individuals]
        else:
            self.individuals = []
        while len(self.individuals) > self.size:
            self.individuals.remove(random.choice(self.individuals))
        while len(self.individuals) < self.size:
            self.individuals.append(self.spawning_pool.spawn())

        # Precomputed values to help to speed up the things a bit
        self.offspring_size = int(math.ceil(size * replacement_rate))
        self.selection_size = len(
            inspect.signature(recombination.perform).parameters
        )

        self.sorted = False
        self.genetic_algorithm = None
        self.sort(init=True)
        self.best_individuals_by_generation = [self[0]]

    def __len__(self):
        """ Returns the number fo individuals this population has. """
        return len(self.individuals)

    def __delitem__(self, i):
        """ Removes the ith individual from the population.

        The population will be sorted by its fitness before deleting.

        :param i: The ith individual to delete.
        """
        self.sort()
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
        self.sorted = False
        self.__setitem__(i, individual)
        individual.population = self

    def insert(self, i, individual):
        """ Ads a new element to the ith position of the population population.

        This call will cause a new sorting of the individuals the next time an
        access is required. This means that is preferable to make all the
        inserts in the population at once instead doing interleaved readings and
        inserts.

        :param i: The position where insert the individual.
        :param individual: The individual to be inserted in the population
        """
        self.sorted = False
        individual.population = self
        self.individuals.insert(i, individual)

    def __getitem__(self, i):
        """ Returns the individual located on the ith position.

        The population will be sorted before accessing to the element so it's
        correct to assume that the individuals are arranged from fittest (i = 0)
        to least fit (n  len(populaton)).

        :param i: The index of the individual to retrieve.
        :return: The individual.
        """
        self.sort()
        return self.individuals[i]

    def sort(self, init=False):
        """ Sorts this population from best to worst individual.

        :param init: If enabled, the fitness to perform will be the implemented
            in "init_perform" of fitness subclass. Is not expected to be used
            other than in initialization time. Defaults to False.
        """
        if not self.sorted:
            self.individuals.sort(
                key=operator.methodcaller('fitness', init=init),
                reverse=True
            )
            self.sorted = True

    def evolve(self):
        """ A step of evolution is made on this population.

        That means that a full cycle of select-recombine-mutate-replace is
        performed, potentially modifying the individuals this population
        contains.
        """
        # First, we generate the offspring given population replacement rate.
        offspring = []
        while len(offspring) < self.offspring_size:
            # Selection
            parents = self.selection(self, self.selection_size)
            # Recombination
            if take_chances(self.p_recombination):
                progeny = self.recombination(*parents)
            else:
                progeny = parents
            individuals_who_fit = min(
                len(progeny),
                self.offspring_size - len(offspring)
            )
            progeny = random.sample(progeny, individuals_who_fit)
            # Mutation
            for individual in progeny:
                if take_chances(self.p_mutation):
                    self.mutation(individual)
            # Add progeny to the offspring
            offspring.extend(progeny)

        # Once offspring is generated, a replace step is performed
        self.replacement(self, offspring)

        # The best individual is extracted and stored just in case is needed
        self.store_best_individual()

    def evolve_mp_no_funciona(self):
        """ A step of evolution is made on this population.

        That means that a full cycle of select-recombine-mutate-replace is
        performed, potentially modifying the individuals this population
        contains.
        """
        num_selections = int(self.offspring_size / self.selection_size) + 1
        # Selection of each tuple of parents
        with ProcessPoolExecutor(max_workers=1) as executor:
            futures = []
            for _ in range(num_selections):
                future = executor.submit(
                    mp_population_select,
                    self,
                    self,
                    self.selection_size
                )
                futures.append(future)
            selected_parents = (f.result() for f in futures)

        # Offspring generation
        offspring = []
        for parents in selected_parents:
            # Recombination
            if take_chances(self.p_recombination):
                progeny = self.recombination(*parents)
            else:
                progeny = parents
            # Mutation
            for i, individual in enumerate(progeny):
                if take_chances(self.p_mutation):
                    progeny[i] = self.mutation(individual)
            # The new offspring is generated
            offspring.extend(progeny)

        # We remove random individuals in case the list is bigger than expected
        if len(offspring) > self.offspring_size:
            offspring = random.sample(offspring, self.offspring_size)

        # Once offspring is generated, a replace step is performed
        self.replacement(self, offspring)

        # The best individual is extracted and stored just in case is needed
        self.store_best_individual()

    def store_best_individual(self):
        current_gen = self.genetic_algorithm.generation
        best_individual = self[0]

        if len(self.best_individuals_by_generation) > current_gen:
            self.best_individuals_by_generation[current_gen] = best_individual
        else:
            self.best_individuals_by_generation.append(best_individual)

    def best(self, g=None):
        """ Returns the best individual for the gth.

        :param g: The generation from where obtain the best individual. If not
            specified, the returned generation will be the last generation.
        :return: The best individual for that generation.
        """
        return self.best_individuals_by_generation[g or -1]


def mp_population_select(arg, **kwarg):
    """ Helper function for multiprocessing selection in population. """
    return Population.selection(*arg, **kwarg)


class Fitness(metaclass=ABCMeta):
    """ Method to estimate how adapted is the individual to the environment. """

    def __call__(self, individual: Individual, init: bool = False) -> float:
        """ Calculates the fitness of the individual.

        This method does some checks and the delegates the computation of the
        fitness to the "perform" method.

        :param individual: The individual to which estimate the adaptation.
        :param init: If this call to fitness is in initialization time. It
            defaults to False.
        :return: A sortable object representing the adaptation of the individual
            to the environment.
        :raises PyneticsError: If the individual is None.
        """
        if individual is None:
            raise PyneticsError('The individual cannot be None')
        elif init:
            return self.init_perform(individual)
        else:
            return self.perform(individual)

    def init_perform(self, individual: Individual) -> float:
        """ Estimates how adapted is the individual at initialization time.

        This is useful in schemas where the fitness while initializing is
        computed in a different way than along the generations.

        Overriding this method can be tricky, specially in a co-evolutionary
        scheme. In this stage of the algorithm (initialization) the populations
        are not sorted, and it's position on its population cannot depend on the
        best of other individuals of other populations (circular dependency).
        Therefore, calling other_population[0] is not an option here.

        The scheme proposed by Mitchell A. et. al. in "A Cooperative
        Coevolutionary Approach to Function Optimization", the initialization
        may be performed by selecting a random individual among the other
        populations instead the best. For this purpose, a random() method in
        Population class is provided.

        The default behavior is to call method "perform" but can be overridden
        to any other behavior if needed.

        :param individual: The individual to which estimate the adaptation.
        :return: A sortable object representing the adaptation of the individual
            to the environment.
        :raises PyneticsError: If genetic algorithm is not set at the moment. It
            is highly probable that co-evolution is being implemented and that
            init_perform method is needed and .
        """
        try:
            return self.perform(individual)
        except AttributeError as e:
            # If genetic_algorithm property is not set at this moment, it's
            # probably because a co-evolution is being implemented and that an
            # init_perform implementation is required.
            msg = '{}. Maybe an init_perform implementation is needed'.format(e)
            raise PyneticsError(msg)

    @abstractmethod
    def perform(self, individual: Individual) -> bool:
        """ Estimates how adapted is the individual.

        Must return something comparable (in order to be sorted with the results
        of the methods for other fitnesses). It's supposed that, the highest the
        fitness value is, the fittest the individual is in the environment.

        :param individual: The individual to which estimate the adaptation.
        :return: A sortable object representing the adaptation of the individual
            to the environment.
        """


class Mutation(metaclass=ABCMeta):
    """ Defines the behaviour of a genetic algorithm mutation operator. """

    def __call__(self, individual: Individual) -> Individual:
        """ Applies the mutation method to the individual.

        :param individual: an individual to mutate.
        :returns: A clone of the individual with a mutation.
        """
        return self.perform(individual)

    @abstractmethod
    def perform(self, individual: Individual) -> Individual:
        """ Implementation of the mutation operation.

        The mutation implementation must be aware of the implementation type.
        Given that not all the implementations are the same, not all the
        mutation operations may work.

        :param individual: The individual to mutate.
        :returns: A new mutated individual.
        """


class Recombination(metaclass=ABCMeta):
    """ Defines the behaviour of a recombination operator.

    A recombination operator takes a set of individuals (i.e. parents) and
    generates a different set of individuals (i.e. offspring) normally with
    aspects derived from their parents.
    """

    # TODO No sé yo si el  *args: Individual estará correcto.
    def __call__(self, *args: Individual) -> Individual:
        """ Applies the recombine method to a sequence of individuals.

        :param args: A list of one or more Individual instances to use as
            parents in the recombination.
        :returns: A sequence of individuals with characteristics of the parents.
        """
        return self.perform(*args)

    @abstractmethod
    def perform(self, *args: Individual):
        """ Implementation of the recombine method.

        The method will always receive a list of Individual instances, and the
        implementation must be aware of the individual types because given that
        not all implementations are the same, not all the crossover operations
        may work.

        :param args: A list of one or more Individual instances to use as
            parents in the recombination.
        :returns: A sequence of individuals with characteristics of the parents.
        """


class Replacement(metaclass=ABCMeta):
    """ Replacement of individuals of the population. """

    # TODO Los test son los que chequean los tamaños de población.
    # TODO ¿No debería devolver la población?
    # TODO Los comentarios dejan un poco que desear
    def __call__(
        self,
        population: Population,
        offspring: Sequence[Individual]
    ):
        """ Performs some checks before applying the replacement method.

        :param population: The population where make the replacement.
        :param offspring: The new population to use as replacement.
        """
        self.perform(population, offspring)

    @abstractmethod
    def perform(self, population: Population, offspring: Sequence[Individual]):
        """ It makes the replacement according to the subclass implementation.

        It is recommended for perform method to return the same

        :param population: The population where make the replacement.
        :param offspring: The new population to use as replacement.
        """


class Selection(metaclass=ABCMeta):
    """ Selection of the fittest individuals among the population.

    The selection method is defined as a class. However, it is enough to provide
    as a selection method a function that receives a village and a number of
    individuals, and returns a sample of individuals of that size from the given
    population.
    """

    def __init__(self, repetable: bool = False):
        """ Initializes this selector.

        :param repetable: If repetition of individuals is allowed. If true,
            there are chances for the same individual be selected again.
            Defaults to False.
        """
        self.repetable = repetable

    def __call__(self, population: Population, n: int) -> Sequence[Individual]:
        """ Makes some checks to the configuration before delegating selection.

        After checking the parameters, the selection is performed by perform
        method.

        :param population: The population from which select the individuals.
        :param n: The number of individuals to return.
        :return: A sequence of individuals.
        :raises PyneticsError: If length of the population is smaller than the
            number of individuals to select and the repetition parameter is set
            to False (i.e. the same Individual cannot be selected twice or more
            times).
        """
        if not self.repetable and len(population) < n:
            raise PyneticsError()
        else:
            return self.perform(population, n)

    @abstractmethod
    def perform(self, population: Population, n: int) -> Sequence[Individual]:
        """ It makes the selection according to the subclass implementation.

        :param population: The population from which select the individuals.
        :param n: The number of individuals to return.
        :return: A sequence of n individuals.
        """


class Catastrophe(metaclass=ABCMeta):
    """ Defines the behaviour of a genetic algorithm catastrophe operator.

    It's expected for this operator to keep track of the ga and know when to act
    since it will be called every step of the algorithm after replacement
    operation.
    """

    def __call__(self, population: Population):
        """ Tries to apply the catastrophic operator to the population.

        This method does some checks and the delegates the application of the
        catastrophic operator to the "perform" method.

        :param population: The population where apply the catastrophic method.
        """
        if population is None:
            raise ValueError('The population cannot be None')
        else:
            return self.perform(population)

    @abstractmethod
    def perform(self, population: Population):
        """ Implementation of the catastrophe operation.

        :param population: the population which may suffer the catastrophe
        """
