import inspect
import math
import multiprocessing
from typing import Callable, Sequence

import random

from pynetics import Population, GeneticAlgorithm, SpawningPool, Replacement, \
    take_chances, PyneticsError, Individual


class SimpleGA(GeneticAlgorithm):
    """ Simple implementation of a GeneticAlgorithm

    This subclass implements the basic behavior of a genetic algorithm with some
    degree of configuration.
    """

    def __init__(
        self,
        stop_condition: Callable[[GeneticAlgorithm], bool],
        size: int,
        spawning_pool: SpawningPool,
        selection: Callable[[Population, int], int],
        recombination: Callable[[Sequence[Individual]], Sequence[Individual]],
        mutation: Callable[[Individual, float], Individual],
        replacement: Replacement,
        p_recombination: float = 0.9,
        p_mutation: float = 0.1,
        replacement_rate: float = 1.0,
    ):
        """ Initializes this instance.

        :param stop_condition: The condition to be met in order to stop the
            genetic algorithm.
        :param size: The size this population should have.
        :param spawning_pool: The object that generates individuals.
        :param selection: The method to select individuals of the population to
            recombine.
        :param replacement: The method that will add and remove individuals from
            the population given the set of old individuals (i.e. the ones on
            the population before the evolution step) and new individuals (i.e.
            the offspring).
        :param recombination: The method to recombine parents in order to
            generate an offspring with characteristics of the parents. If none,
            no recombination will be applied.
        :param mutation: The method to mutate an individual. If none, no
            mutation over the individual will be applied. If not provided, no
            mutation is performed.
        :param p_recombination: The odds for recombination method to be
            performed over a set of selected individuals to generate progeny. If
            not performed, progeny will be the parents. Must be a value between
            0 and 1 (both included). If not provided, defaults to 1.0.
        :param p_mutation: The odds for mutation method to be performed over a
            progeny. It's applied once for each individual. If not performed the
            individuals will not be modified. Must be a value between 0 and 1
            (both included). If not provided, it defaults to 0.0 (no mutation is
            performed).
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
        super().__init__(stop_condition=stop_condition)

        self.init_population = Population(
            size=size,
            spawning_pool=spawning_pool,
        )
        self.offspring_size = int(math.ceil(size * replacement_rate))
        self.selection = selection
        self.recombination = recombination
        self.mutation = mutation
        self.replacement = replacement
        self.replacement_rate = replacement_rate
        self.p_recombination = p_recombination
        self.p_mutation = p_mutation

        self.selection_size = len(
            inspect.signature(recombination.__call__).parameters
        )
        self.population = None
        self.best_individuals = None

    def initialize(self):
        self.population = self.init_population
        self.best_individuals = [self.population.best()]

    def step(self):
        offspring = []
        while len(offspring) < self.offspring_size:
            # Selection
            parents = self.selection(self.population, self.selection_size)
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
            mutated_progeny = [
                self.mutation(individual, self.p_mutation)
                for individual in progeny
                ]
            # Add progeny to the offspring
            offspring.extend(mutated_progeny)

        # Once offspring is generated, a replace step is performed
        self.replacement(self.population, offspring)

        # We store the best individual for further information
        if self.generation < len(self.best_individuals):
            self.best_individuals[self.generation] = self.population.best()
        else:
            self.best_individuals.append(self.population.best())

    def best(self, generation=None):
        generation = generation or -1
        if generation > len(self.best_individuals) - 1:
            raise PyneticsError()
        else:
            return self.best_individuals[generation]


class ConcurrentGA(GeneticAlgorithm):
    def __init__(
        self,
        stop_condition: Callable[[GeneticAlgorithm], bool],
        size: int,
        spawning_pool: SpawningPool,
        selection: Callable[[Population, int], int],
        recombination: Callable[[Sequence[Individual]], Sequence[Individual]],
        mutation: Callable[[Individual, float], Individual],
        replacement: Replacement,
        p_recombination: float = 0.9,
        p_mutation: float = 0.1,
        replacement_rate: float = 1.0,
        processes: int = None
    ):
        super().__init__(stop_condition=stop_condition)

        self.spawning_pool = spawning_pool
        self.offspring_size = int(math.ceil(size * replacement_rate))
        self.selection = selection
        self.recombination = recombination
        self.mutation = mutation
        self.replacement = replacement
        self.replacement_rate = replacement_rate
        self.p_recombination = p_recombination
        self.p_mutation = p_mutation
        self.nproc = processes or multiprocessing.cpu_count()
        # Round the population size to the next multiple of the # of processes
        self.psize = (size + self.nproc - 1) // self.nproc * self.nproc
        # Population will be splited in chunks of psize / nproc
        self.csize = self.psize / self.nproc

        self.selection_size = len(
            inspect.signature(recombination.__call__).parameters
        )
        self.population = None
        self.best_individuals = None

    def initialize(self):
        self.population = Population(
            size=self.psize,
            spawning_pool=self.spawning_pool,
        )
        self.best_individuals = [self.population.best()]

    def step(self):
        random.shuffle(self.population)
        populations = [
            Population(
                size=self.psize,
                spawning_pool=self.spawning_pool,
                individuals=self.population[self.csize * i:self.csize * (i + 1)]
            ) for i in self.nproc
            ]

        offsprings_queue = multiprocessing.Queue()
        processes = []
        for i, population in enumerate(populations):
            p = multiprocessing.Process(
                target=self.evolve,
                args=(population, offsprings_queue))
            processes.append(p)
            p.start()

        offspring = []
        for _ in range(self.nproc):
            offspring.extend(offsprings_queue.get())
        for process in processes:
            process.join()

        self.population = Population(
            size=self.psize,
            spawning_pool=self.spawning_pool,
            individuals=offspring,
        )

        # We store the best individual for further information
        if self.generation < len(self.best_individuals):
            self.best_individuals[self.generation] = self.population.best()
        else:
            self.best_individuals.append(self.population.best())

    def evolve(self, population, offspring_queue):
        offspring = []
        for i in range(1):
            offspring = []
            while len(offspring) < self.offspring_size:
                # Selection
                parents = self.selection(population, self.selection_size)
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
                mutated_progeny = [
                    self.mutation(individual, self.p_mutation)
                    for individual in progeny
                    ]
                # Add progeny to the offspring
                offspring.extend(mutated_progeny)

            # Once offspring is generated, a replace step is performed
            self.replacement(population, offspring)

        # We add to the queue the result and we're finished
        offspring_queue.put(offspring)


    def best(self, generation=None):
        generation = generation or -1
        if generation > len(self.best_individuals) - 1:
            raise PyneticsError()
        else:
            return self.best_individuals[generation]


def mp_factorizer(nums, nprocs):
    def worker(nums, out_q):
        """ The worker function, invoked in a process. 'nums' is a
            list of numbers to factor. The results are placed in
            a dictionary that's pushed to a queue.
        """
        outdict = {}
        for n in nums:
            outdict[n] = factorize_naive(n)
        out_q.put(outdict)

