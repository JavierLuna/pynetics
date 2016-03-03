import time

from pynetics import Fitness, Population
from pynetics.algorithms import GeneticAlgorithm
from pynetics.catastrophe import NoCatastrophe
from pynetics.ga_list import RandomMaskRecombination, OnePointRecombination, \
    RandomGeneValue, \
    TwoPointRecombination
from pynetics.ga_list.ga_bin import BinaryIndividualSpawningPool, binary_alleles
from pynetics.replacements import HighElitism
from pynetics.selections import Tournament
from pynetics.stop import FitnessBound

population_size = 20000
tournament = 3
replacement_rate = 0.99
individual_size = 100


class MaximizeOnesFitness(Fitness):
    """ Fitness where more 1's implies higher fitness. """

    def perform(self, individual):
        return 1. / (1. + (len(individual) - sum(individual)))


if __name__ == '__main__':
    ga = GeneticAlgorithm(
        FitnessBound(1),
        [
            Population(
                name='1',
                size=population_size,
                replacement_rate=replacement_rate,
                spawning_pool=BinaryIndividualSpawningPool(
                    individual_size,
                    fitness=MaximizeOnesFitness()
                ),
                selection=Tournament(tournament),
                recombination=RandomMaskRecombination(),
                p_recombination=1,
                mutation=RandomGeneValue(binary_alleles),
                p_mutation=0.1,
                replacement=HighElitism(),
            )
        ],
        NoCatastrophe(),
    )

    #print([' ' if x == 0 else '#' for x in ga.populations[0].best()])
    '''
    ga.listeners[GeneticAlgorithm.MSG_STEP_FINISHED].append(
        lambda g: print(
            '{}\t->\t{} ({})'.format(
                ga.generation,
                ''.join(
                    [' ' if x == 0 else '#' for x in ga.populations[0].best()]
                ),
                ga.populations[0].best().fitness()
            )
        )
    )
    '''


    class Clock:
        def __init__(self):
            self.start_time = None

        def start(self):
            self.start_time = time.time()

        def end(self):
            print(time.time() - self.start_time)


    clock = Clock()
    ga.listeners[GeneticAlgorithm.MSG_STEP_STARTED].append(lambda g: clock.start())
    ga.listeners[GeneticAlgorithm.MSG_STEP_FINISHED].append(lambda g: clock.end())
    ga.run()
    #print([' ' if x == 0 else '#' for x in ga.populations[0].best()])
