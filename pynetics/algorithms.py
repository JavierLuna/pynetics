import copy
import enum
import inspect
import math

from pynetics import Population, PyneticsError, NoMutation


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

    def __init__(
            self,
            stop_condition: Callable[['GeneticAlgorithm'], bool]
    ):
        """ Initializes this object.

        It expects a criteria to be met in order to stop the algorithm.

        :param stop_condition: The criteria to be met after each of the steps
            performed by the algorithm to stop it. It could be any callable
            while complying the contract. The superclass
            :class:`stop.StopCondition` is compliant with the contract.
        """
        self.stop_condition = stop_condition
        self.listeners = collections.defaultdict(list)
        self.generation = 0

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

    def step(self):
        """ Performs an step in the genetic algorithm.

        Depending on the implementation, this step may be very different so
        there is no general implementation here.
        """

    @abc.abstractmethod
    def best(self, generation=None) -> 'Individual':
        """ The best individual obtained.

        :param generation: The generation of the individual that we want to
            recover. If not set, this will be the one emerged in the last
            generation. Defaults to None (not set, thus last generation).
        :return: The best individual generated in the specified generation or in
            the last one if no generation is specified.
        """

    @abc.abstractmethod
    def clone(self):
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

    def on_start(self, f):
        """ Specifies a functor to be called when the algorithm starts.

        This function will be called AFTER initialization but BEFORE the first
        iteration, including the check against the stop condition.

        :param f: The functor to be called. It must accept a GeneticAlgorithm
            instance as a parameter.
        """
        ga = self.clone()
        ga.listeners[Event.ALGORITHM_START].append(f)
        return ga

    def on_end(self, f):
        """ Specifies a functor to be called when the algorithm ends.

        Particularly, this method will be called AFTER the stop condition
        has been met.

        :param f: The functor to be called. It must accept a GeneticAlgorithm
            instance as a parameter.
        """
        ga = self.clone()
        ga.listeners[Event.ALGORITHM_END].append(f)
        return ga

    def on_step_start(self, f):
        """ Specifies a functor to be called when an iteration step starts.

        This method will be called AFTER the stop condition has been checked
        and proved to be false) and BEFORE the new step is computed.

        :param f: The functor to be called. It must accept a GeneticAlgorithm
            instance as a parameter.
        """
        ga = self.clone()
        ga.listeners[Event.STEP_START].append(f)
        return ga

    def on_step_end(self, f):
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

    def clone(self):
        ga = super().clone()
        # Copying all the numeric values
        ga.population_size = self.population_size
        ga.offspring_size = self.offspring_size
        ga.replacement_rate = self.replacement_rate
        ga.p_recombination = self.p_recombination
        ga.p_mutation = self.p_mutation
        ga.selection_size = self.selection_size
        # Deep copying the objects (just in case they're maintaining any state
        ga.spawning_pool = copy.deepcopy(self.spawning_pool)
        ga.fitness = copy.deepcopy(self.fitness)
        ga.selection = copy.deepcopy(self.selection)
        ga.recombination = copy.deepcopy(self.recombination)
        ga.mutation = copy.deepcopy(self.mutation)
        ga.diversity = copy.deepcopy(self.diversity)
        ga.replacement = copy.deepcopy(self.replacement)
        ga.population = copy.deepcopy(self.population)
        ga.best_individuals = copy.deepcopy(self.best_individuals)

        return ga

    def initialize(self):
        super().initialize()
        # Generate a new population
        self.population = Population(
            size=self.population_size,
            spawning_pool=self.spawning_pool,
        )
        # Clear the best individuals historical cache
        self.best_individuals.clear()

    def best(self, generation=None):
        if self.best_individuals:
            generation = generation or -1
            if generation > len(self.best_individuals) - 1:
                raise PyneticsError()
            else:
                return self.best_individuals[generation]
        else:
            return None
