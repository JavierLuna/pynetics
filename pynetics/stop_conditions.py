import abc


class StopCondition(metaclass=abc.ABCMeta):
    """ A condition to be met in order to stop the algorithm.

    Although the stop condition is defined as a class, it's enough to provide a
    function that is able to discern whether the time has come to stop (True or
    False) receiving as parameter the population.
    """

    @abc.abstractmethod
    def __call__(self, populations):
        """ Checks if this stop condition is met.

        :param populations: The list of populations being evolved in the ga.
        :return: True if criteria is met, false otherwise.
        """


class StepsNumStopCondition(StopCondition):
    """ If the genetic algorithm has made enough iterations. """

    def __init__(self, steps):
        """ Initializes this function with the number of iterations.

        :param steps: An integer value.
        """
        self.__steps = steps

    def __call__(self, populations):
        """ Checks if this stop criteria is met.

        It will look at the generation of the populations. It's expected that
        all the populations in the same problem have the same generation at the
        same time so the algorithm will check against the first. If its
        generation is greater or equal to the specified in initialization
        method, the criteria is met.

        :param populations: The genetic algorithm to check.
        :return: True if criteria is met, false otherwise.
        """
        return populations[0].generation >= self.__steps
