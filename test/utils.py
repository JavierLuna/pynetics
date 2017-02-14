from random import choice

from pynetics import StopCondition, Individual, SpawningPool, Fitness, Mutation, \
    Recombination, Replacement, Selection, Diversity
from pynetics.ga_list import Alleles


class DummyIndividual(Individual):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def phenotype(self):
        return self.name

    def clone(self) -> 'Individual':
        new_i = super().clone()
        new_i.name = self.name
        return new_i


class ConstantFitness(Fitness):
    def __init__(self, fitness):
        super().__init__()
        self.fitness = fitness

    def __call__(self, individual: 'Individual') -> float:
        return self.fitness


class ConstantDiversity(Diversity):
    def __init__(self, diversity):
        super().__init__()
        self.diversity = diversity

    def __call__(self, individuals):
        return self.diversity


class DummyStopCondition(StopCondition):
    def __call__(self, genetic_algorithm):
        return True


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


def individuals(n, fitness_method=ConstantFitness(0.5)):
    result = []
    for i in range(n):
        individual = DummyIndividual()
        individual.fitness_cached = i
        individual.fitness_method = fitness_method
        result.append(individual)
    return result
