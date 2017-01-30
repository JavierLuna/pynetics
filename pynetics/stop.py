from pynetics import StopCondition, GeneticAlgorithm


class StepsNum(StopCondition):
    """ If the genetic algorithm has made enough iterations. """

    def __init__(self, steps: int):
        """ Initializes this function with the number of iterations.

        :param steps: The number of iterations to do before stop.
        """
        self.steps = steps

    def __call__(self, ga: GeneticAlgorithm) -> bool:
        """ Checks if this stop criteria is met.

        It will look at the generation of the genetic algorithm. It's expected
        that, if its generation is greater or equal than the specified in
        initialization method, the criteria is met.

        :param ga: The genetic algorithm where this stop condition belongs.
        :return: True if criteria is met, false otherwise.
        """
        return ga.generation >= self.steps


class FitnessBound(StopCondition):
    """ If the genetic algorithm obtained a fine enough individual. """

    def __init__(self, fitness_bound: float):
        """ Initializes this function with the upper bound for the fitness.

        :param fitness_bound: A fitness value. The criteria will be met when the
            fitness in the algorithm (in one or all populations managed, see
            below) is greater than this specified fitness.
        """
        self.fitness_bound = fitness_bound

    def __call__(self, ga: GeneticAlgorithm) -> bool:
        """ Checks if this stop criteria is met.

        It will look at the fitness of the best individual the genetic algorithm
        has discovered. In case of its fitness being greater or equal than the
        specified at initialization time, the condition will be met and the
        algorithm will stop.

        :param ga: The genetic algorithm where this stop condition belongs.
        :return: True if criteria is met, false otherwise.
        """
        return ga.best().fitness() >= self.fitness_bound
