import random
from array import array
from typing import Callable

from pynetics import Individual, Mutation, take_chances, Diversity
from pynetics.ga_list import ListRecombination, ListIndividualSpawningPool, \
    FiniteSetAlleles


class BinaryIndividual(Individual):
    """ An individual represented by a binary chromosome. """

    def fitness(self) -> float:
        pass

    def __init__(self):
        super().__init__()
        self.genes = []

    def __getitem__(self, index):
        return self.genes[index]

    def __delitem__(self, index):
        return self.genes.remove(index)

    def insert(self, index, value):
        self.genes.insert(index, value)

    def __setitem__(self, index, value):
        self.genes[index] = value

    def __len__(self):
        return len(self.genes)

    def phenotype(self):
        return self.genes

    def clone(self):
        clone = super().clone()
        clone.genes = self.genes[:]
        return clone

    def __str__(self):
        return ''.join(str(b) for b in self.genes)


class BinaryAlleles(FiniteSetAlleles):
    """ Possible alleles are 0 or 1. """

    def __init__(self):
        super().__init__((0, 1))


class BinaryIndividualSpawningPool(ListIndividualSpawningPool):
    """ Defines the methods for creating binary individuals. """

    def __init__(self, fitness: Callable[[Individual], float], size: int):
        """ Initializes this spawning pool for generating binary individuals.

        :param fitness: The fitness the individuals will have in order to be
            evaluated.
        :param size: The size of the individuals to be created from
            this spawning pool.
        """
        super().__init__(fitness=fitness)
        self.individual_size = size

    def create(self) -> BinaryIndividual:
        individual = BinaryIndividual()
        individual.genes = array('B', [
            random.getrandbits(1)
            for _ in range(self.individual_size)
            ])
        return individual


class MomentOfInertia(Diversity):
    """ A diversity implementation based on centroids and inertia.

    Extracted from paper "Measurement of Population Diversity" of R.W. Morrison
    et. al.
    """

    # TODO Not working. Review.
    def __call__(self, individuals):
        centroid_vector = [0 for _ in individuals[0]]
        for individual in individuals:
            for i, gene in enumerate(individual):
                centroid_vector[i] += gene

        for i, centroid in enumerate(centroid_vector):
            centroid_vector[i] = centroid / len(individuals)

        diversity = 0
        for i in range(len(individuals[0])):
            for j, individual in enumerate(individuals):
                diversity += (individual[i] - centroid_vector[i]) ** 2
        return diversity


class AverageHamming(Diversity):
    """ Diversity implementation of the average of each hamming loci distances.

    Predicting Convergence Time for Genetic Algorithms Sushil J. Louis and
    Gregory J. E. Rawlins
    """

    def __call__(self, individuals):
        diversity = 0.0
        total = 0.0
        individual_len = len(individuals[0])
        for j, i1 in enumerate(individuals[:-1]):
            for k, i2 in enumerate(individuals[j:]):
                diversity += self.f(i1, i2)
                total += individual_len
        return diversity / total

    @staticmethod
    def f(i1, i2):
        hamming = 0.0
        for g1, g2 in zip(i1, i2):
            if g1 != g2:
                hamming += 1
        return hamming


class GeneralizedRecombination(ListRecombination):
    """ Offspring is obtained by crossing individuals as they where integers.

    NOTE: Works only for individuals with list chromosomes of binary alleles.
    Ok, may work for other individuals with list chromosomes, but the results
    may be strange (perhaps better, but I doubt it)
    """

    def __call__(self, parent1, parent2):
        """ Applies the crossover operator.

        :param parent1: One of the individuals from which generate the progeny.
        :param parent2: The other.
        :return: A tuple with the two children for this parents.
        """
        child1, child2 = super().__call__(parent1, parent2)
        # Obtain the crossover range (as integer values)
        a = int(''.join([str(b1 & b2) for (b1, b2) in zip(child1, child2)]), 2)
        b = int(''.join([str(b1 | b2) for (b1, b2) in zip(child1, child2)]), 2)

        # Get the children (as integer values)
        c = random.randint(a, b)
        d = b - (c - a)

        # Convert to binary lists again (NOTE: we remove the starting 0b)
        bin_formatter = '{:#0' + str(len(child1) + 2) + 'b}'
        bin_c = [int(x) for x in bin_formatter.format(c)[2:]]
        bin_d = [int(x) for x in bin_formatter.format(d)[2:]]

        # Convert to chromosomes and we're finish
        for i in range(len(bin_c)):
            child1[i], child2[i] = bin_c[i], bin_d[i]
        return child1, child2


class AllGenesCanSwitch(Mutation):
    def __call__(self, individual, p):
        """ Returns the same instance of the individual mutated.

        :param individual: The individual to mutate.
        :param p: The probability for a gene to mutate.
        :return: The same instance maybe mutated).
        """
        for i, gene_value in enumerate(individual):
            if take_chances(probability=p):
                individual[i] = 1 - gene_value
        return individual
