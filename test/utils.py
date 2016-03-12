from random import choice

from pynetics import StopCondition, Individual, SpawningPool, Fitness, Mutation, \
    Recombination, Replacement, Selection, Population
from pynetics.ga_list import Alleles


class DummyStopCondition(StopCondition):
    def __call__(self, genetic_algorithm):
        return True


class DummyIndividual(Individual):
    def phenotype(self):
        return 'DummyIndividual'

    def clone(self):
        individual = type(self)()
        individual.__dict__.update(self.__dict__)
        return individual


class DummySpawningPool(SpawningPool):
    def create(self):
        return DummyIndividual()


class DummyFitness(Fitness):
    def perform(self, individual):
        return 0.5


class DummyMutation(Mutation):
    def __call__(self, individual, p):
        return individual.clone()


class DummyRecombination(Recombination):
    def __call__(self, *args):
        return [i.clone() for i in args]


class DummyReplacement(Replacement):
    def __call__(self, population, individual):
        return population


class DummySelection(Selection):
    def perform(self, population, n):
        return population[:n]


class DummyCatastrophe(Selection):
    def perform(self, population, n):
        return population[:n]


class DummyAlleles(Alleles):
    def get(self):
        return choice(('D', 'U', 'M', 'M', 'Y'))


class DummyPopulation(Population):
    def __init__(self, size, spawning_pool=None, individuals=None):
        super().__init__(
            size=size,
            spawning_pool=spawning_pool or DummySpawningPool(
                fitness=DummyFitness()
            ),
            individuals=individuals or [],
        )


def individuals(n, fitness_method=DummyFitness()):
    result = []
    for i in range(n):
        individual = DummyIndividual()
        individual.fitness_cached = i
        individual.fitness_method = fitness_method
        result.append(individual)
    return result
