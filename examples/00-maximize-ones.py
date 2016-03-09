import time

from pynetics import Fitness
from pynetics.algorithms import SimpleGA
from pynetics.ga_bin import BinaryIndividualSpawningPool, binary_alleles
from pynetics.ga_list import RandomMaskRecombination, NGeneRandomValue
from pynetics.replacements import LowElitism
from pynetics.selections import Tournament
from pynetics.stop import FitnessBound

population_size = 100
tournament = 5
replacement_rate = 0.99
individual_size = 1000


class MaximizeOnesFitness(Fitness):
    """ Fitness where more 1's implies higher fitness. """

    def perform(self, individual):
        # for _ in range(1000):
        #    fitness = 1. / (1. + (len(individual) - sum(individual)))
        return 1. / (1. + (len(individual) - sum(individual)))


class Listener(SimpleGA.GAListener):
    def __init__(self):
        self.start_time = None

    def step_finished(self, ga):
        print('generation: {}\tfitness: {:.5f}\ttime: {:.5f}'.format(
            ga.generation,
            ga.best().fitness(),
            time.time() - self.start_time,
            # ''.join([' ' if not g else '#' for g in ga.best()])
        ))

    def step_started(self, ga):
        self.start_time = time.time()

    def algorithm_finished(self, ga):
        pass

    def algorithm_started(self, ga):
        pass


if __name__ == '__main__':
    ga = SimpleGA(
        stop_condition=FitnessBound(1),
        size=population_size,
        spawning_pool=BinaryIndividualSpawningPool(
            individual_size,
            fitness=MaximizeOnesFitness()
        ),
        selection=Tournament(tournament),
        recombination=RandomMaskRecombination(),
        mutation=NGeneRandomValue(binary_alleles),
        replacement=LowElitism(),
        p_recombination=1,
        p_mutation=.1,
        replacement_rate=replacement_rate,
    )
    ga.listeners.append(Listener())
    ga.run()
