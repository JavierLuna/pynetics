from typing import Sequence

from . import Population, Individual, Replacement


class LowElitism(Replacement):
    """ Low elitism replacement.

    The method will replace the less fit individuals by the ones specified in
    the offspring. This makes this operator elitist, but at least not much.
    Moreover, if offspring size equals to the population size then it's a full
    replacement (i.e. a generational scheme).

    This method guarantees that the population size remains unchanged after the
    replacement.
    """

    def perform(
            self, *,
            population: Population,
            offspring: Sequence[Individual]
    ):
        """ Removes less fit individuals and then inserts the offspring.

        :param population: The population where make the replacement.
        :param offspring: The new population to use as replacement.
        """
        if offspring:
            population.sort()
            del population[-len(offspring):]
            population.extend(offspring)


class HighElitism(Replacement):
    """ Drops the less fit individuals among all (population plus offspring).

    The method will add all the individuals in the offspring to the population,
    removing afterwards those individuals less fit. This makes this operator
    highly elitist but if length os population and offspring are the same, the
    process will result in a full replacement, i.e. a generational scheme of
    replacement.

    This method guarantees that the population size remains unchanged after the
    replacement.
    """

    def perform(
            self, *,
            population: Population,
            offspring: Sequence[Individual]
    ):
        """ Inserts the offspring in the population and removes the less fit.

        :param population: The population where make the replacement.
        :param offspring: The new population to use as replacement.
        """
        if offspring:
            population.extend(offspring)
            population.sort()
            del population[-len(offspring):]
