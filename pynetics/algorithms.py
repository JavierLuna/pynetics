import inspect
import math
import random

from pynetics import Population, GeneticAlgorithm, take_chances, PyneticsError, \
    NoMutation


class SimpleGA(GeneticAlgorithm):
    """ Simple implementation of a GeneticAlgorithm

    This subclass implements the basic behavior of a genetic algorithm with some
    degree of configuration.
    """

    def __init__(
            self,
            stop_condition,
            population_size,
            spawning_pool,
            fitness,
            selection,
            recombination,
            replacement,
            mutation=None,
            diversity=None,
            p_recombination=0.9,
            p_mutation=0.1,
            replacement_rate=1.0,
    ):
        """ Initializes this instance.

        :param stop_condition: The condition to be met in order to stop the
            genetic algorithm.
        :param population_size: The size this population should have.
        :param spawning_pool: The object that generates individuals.
        :param fitness: The method to evaluate individuals. It's expected to be
            a callable that returns a float value where the higher the value,
            the better the individual. Instances of subclasses of class Fitness
            can be used for this purpose.
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
        :param diversity: The method to compute the diversity of a sequence of
            individuals generated by this SpawningPool instance. Is expected to
            be a function that generates a diversity representation given a
            subset of individuals. Instances of subclasses of class Diversity
            can be used for this purpose.
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

        self.population_size = population_size
        self.spawning_pool = spawning_pool
        self.fitness = fitness
        self.offspring_size = int(math.ceil(population_size * replacement_rate))
        self.selection = selection
        self.recombination = recombination
        self.mutation = mutation or NoMutation()
        self.diversity = diversity
        self.replacement = replacement
        self.replacement_rate = replacement_rate
        self.p_recombination = p_recombination
        self.p_mutation = p_mutation

        self.selection_size = len(
            inspect.signature(recombination.__call__).parameters
        )
        self.population = None
        self.best_individuals = []

    def initialize(self):
        super().initialize()
        # Generate a new population
        self.population = Population(
            size=self.population_size,
            spawning_pool=self.spawning_pool,
        )
        for individual in self.population:
            individual.fitness_method = self.fitness
        # Clear the best individuals historical cache
        self.best_individuals.clear()

    def step(self):
        offspring = []
        while len(offspring) < self.offspring_size:
            # Selection
            parents = self.selection(self.population, self.selection_size)
            # Recombination
            if take_chances(self.p_recombination):
                progeny = self.recombination(*parents)
            else:
                progeny = [i.clone() for i in parents]
            # Mutation
            individuals_who_fit = min(
                len(progeny),
                self.offspring_size - len(offspring)
            )
            progeny = [
                self.mutation(individual, self.p_mutation)
                for individual in random.sample(progeny, individuals_who_fit)
                ]
            # Add progeny to the offspring
            offspring.extend(progeny)

        # Once offspring is generated, a replace step is performed
        self.replacement(self.population, offspring)

        # We store the best individual for further information
        if self.generation < len(self.best_individuals):
            self.best_individuals[self.generation] = self.population.best()
        else:
            self.best_individuals.append(self.population.best())

    def best(self, generation=None):
        if self.best_individuals:
            generation = generation or -1
            if generation > len(self.best_individuals) - 1:
                raise PyneticsError()
            else:
                return self.best_individuals[generation]
        else:
            return None
