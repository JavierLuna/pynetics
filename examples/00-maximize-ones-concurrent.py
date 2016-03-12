import time

from pynetics.algorithms import SimpleGA, ConcurrentGA
from pynetics.ga_bin import BinaryIndividualSpawningPool, AllGenesCanSwitch, \
    MomentOfIntertiaDiversity
from pynetics.ga_list import RandomMaskRecombination
from pynetics.replacements import LowElitism
from pynetics.selections import Tournament
from pynetics.stop import FitnessBound

population_size = 10
tournament = 3
replacement_rate = 0.9
individual_size = 50


def maximize_ones_fitness(individual):
    return 1. / (1. + (len(individual) - sum(individual)))


class Listener(SimpleGA.GAListener):
    def __init__(self):
        self.start_time = 0
        self.total_time = 0

    def step_finished(self, ga):
        end_time = time.time() - self.start_time
        print(
            'generation: {}\tfitness: {:.2f}\tdiversity: {:.2f}\ttime: {:.5f}\tIndividual: {}'.format(
                ga.generation,
                ga.best().fitness(),
                ga.best().population.diversity(),
                end_time,
                ''.join([' ' if not g else '#' for g in ga.best()]),
            ))
        self.total_time += end_time

    def step_started(self, ga):
        self.start_time = time.time()

    def algorithm_finished(self, ga):
        print('Total time: {} s.'.format(self.total_time))

    def algorithm_started(self, ga):
        self.start_time = 0
        self.total_time = 0


if __name__ == '__main__':
    ga = ConcurrentGA(
        stop_condition=FitnessBound(1),
        size=population_size,
        spawning_pool=BinaryIndividualSpawningPool(
            size=individual_size,
            fitness=maximize_ones_fitness,
            diversity=MomentOfIntertiaDiversity(),
        ),
        selection=Tournament(tournament),
        recombination=RandomMaskRecombination(),
        mutation=AllGenesCanSwitch(),
        replacement=LowElitism(),
        p_recombination=1,
        p_mutation=1. / individual_size,
        replacement_rate=replacement_rate,
    )
    ga.listeners.append(Listener())
    ga.run()
