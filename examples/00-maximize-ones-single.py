import time

from pynetics.algorithms import SimpleGA
from pynetics.ga_bin import BinaryIndividualSpawningPool, AllGenesCanSwitch, \
    AverageHamming
from pynetics.ga_list import RandomMaskRecombination
from pynetics.replacements import LowElitism
from pynetics.selections import Tournament
from pynetics.stop import FitnessBound

population_size = 10
tournament = 3
replacement_rate = 0.9
individual_size = 100


def maximize_ones_fitness(individual):
    return 1. / (1. + (len(individual) - sum(individual)))

if __name__ == '__main__':
    ga = SimpleGA(
        stop_condition=FitnessBound(1),
        population_size=population_size,
        spawning_pool=BinaryIndividualSpawningPool(size=individual_size),
        fitness=maximize_ones_fitness,
        diversity=AverageHamming(),
        selection=Tournament(tournament),
        recombination=RandomMaskRecombination(),
        mutation=AllGenesCanSwitch(),
        replacement=LowElitism(),
        p_recombination=1,
        p_mutation=1. / individual_size,
        replacement_rate=replacement_rate,
    ).on_step_end(
        lambda genetic_algorithm: print(
            'generation: {}\tfitness: {:.2f}\tIndividual: {}'.format(
                genetic_algorithm.generation,
                genetic_algorithm.best().fitness(),
                genetic_algorithm.best(),
            )
        )
    )
    ga.run()
