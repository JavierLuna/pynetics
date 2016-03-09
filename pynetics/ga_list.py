from typing import Any, Sequence

import random
from abc import ABCMeta, abstractmethod

from pynetics import SpawningPool, Fitness, Individual, Recombination, \
    PyneticsError, take_chances, Mutation


class Alleles(metaclass=ABCMeta):
    """ The alleles are all the possible values a gene can take. """

    @abstractmethod
    def get(self) -> Any:
        """ Returns a random value of all the possible existent values. """


class FiniteSetAlleles(Alleles):
    """ The possible alleles belong to a finite set of symbols. """

    def __init__(self, symbols: Sequence[Any]):
        """ Initializes this set of alleles with its sequence of symbols.

        The duplicated values are removed in the list maintained by this alleles
        class, so none of the different symbols has a higher probability to be
        selected.

        :param symbols: The sequence of symbols.
        """
        self.symbols = set(symbols)

    def get(self):
        """ A random value is selected uniformly over the set of values. """
        return random.choice(tuple(self.symbols))


class ListIndividualSpawningPool(SpawningPool, metaclass=ABCMeta):
    """ Defines the methods for creating individuals required by population. """

    def __init__(self, size: int, alleles: Alleles, fitness: Fitness):
        """ Initializes this spawning pool for generating list individuals.

        :param size: The size of the individuals to be created from this
            spawning pool.
        :param alleles: The alleles to be used as values of the genes.
        :param fitness: The fitness to use to evaluate the individuals generated
            by this SpawningPool instance.
        """
        super().__init__(fitness=fitness)
        self.size = size
        self.alleles = alleles

    def create(self):
        """ Creates a new individual randomly.

        :return: A new Individual object.
        """
        individual = ListIndividual()
        for _ in range(self.size):
            individual.append(self.alleles.get())
        return individual


# Maybe instead inherit from list is better inherit from mutablesequence
class ListIndividual(Individual, list):
    """ An individual whose representation is a list of finite values. """

    def __eq__(self, individual: 'ListIndividual') -> bool:
        """ The equality between two list individuals is True if they:

        1. Have the same length
        2. Any two genes in the same position have the same value.
        """
        return len(self) == len(individual) and all(
            [x == y for (x, y) in zip(self, individual)]
        )

    def phenotype(self):
        """ A default phenotype for this kind of invdividuals.

        :return: A list where each of the elements is the string representation
            of each of the genes.
        """
        return [str(g) for g in self]

    def clone(self):
        """ Clones this ListIndividual.

        :return: A ListIndividual looking exactly like this.
        """
        individual = super().clone()
        for gene in self:
            individual.append(gene)
        return individual


class FixedLengthListRecombination(Recombination, metaclass=ABCMeta):
    """ Behavior for recombinations where lengths should be the same. """

    @abstractmethod
    def __call__(self, *args):
        """ Performs checks over the lengths before executing perform.

        Specifically, it checks if the length of all individuals are the same.
        In so, the crossover operation is done. If not, a PyneticsError is
        raised.

        :param args: The individuals to use as parents from which generate
            the progeny.
        :return: A tuple with cloned parents (same order).
        :raises PyneticsError: If not all the individuals has the same length.
        """
        lengths = [len(i) for i in args]
        if not lengths.count(lengths[0]) == len(lengths):
            raise PyneticsError('Individuals must have the same length')
        else:
            return tuple(i.clone() for i in args)


class OnePointRecombination(FixedLengthListRecombination):
    """ Offspring is created by mixing the parents using one random pivot point.

    This recombination implementation works with two (and only two) individuals
    of type ListIndividual (or subclasses).
    """

    def __call__(self, parent1: ListIndividual, parent2: ListIndividual):
        """ Offspring is obtained mixing the parents with one pivot point.

        One example:

        parents  : aaaaaaaa, bbbbbbbb
        pivot    : 3
        -----------
        children : aaabbbbb, bbbaaaaa

        :param parent1: One of the individuals from which generate the progeny.
        :param parent2: The other.
        :return: A list of two individuals, each a child containing some
            characteristics from their parents.
        """
        child1, child2 = super().__call__(parent1, parent2)

        p = random.randint(1, len(parent1) - 1)
        for i in range(len(parent1)):
            if i >= p:
                child1[i], child2[i] = parent2[i], parent1[i]
        return child1, child2


class TwoPointRecombination(FixedLengthListRecombination):
    """ Offspring is created by mixing the parents using two random pivot point.

    This crossover implementation works with two (and only two) individuals of
    type ListIndividual (or subclasses).
    """

    def __call__(self, parent1, parent2):
        """ Offspring is obtained mixing the parents with two pivot point.

        One example:

        parents  : aaaaaaaa, bbbbbbbb
        pivot    : 3, 5
        -----------
        children : aaabbaaa, bbbaabbb

        :param parent1: One of the individuals from which generate the progeny.
        :param parent2: The other.
        :return: A list of two individuals, each a child containing some
            characteristics from their parents.
        """
        child1, child2 = super().__call__(parent1, parent2)

        pivots = random.sample(range(len(parent1) - 1), 2)
        p, q = min(pivots[0], pivots[1]), max(pivots[0], pivots[1])
        for i in range(len(parent1)):
            if not p < i < q:
                child1[i], child2[i] = parent2[i], parent1[i]
        return child1, child2


class RandomMaskRecombination(FixedLengthListRecombination):
    """ Offspring is created by using a random mask.

    This crossover implementation works with two (and only two) individuals of
    type ListIndividual (or subclasses).
    """

    def __call__(self, parent1, parent2):
        """ Offspring is obtained generating a random mask.

        This mask determines which genes of each of the progenitors are used on
        each of the the genes. For example:

        parents     : aaaaaaaa, bbbbbbbb
        random mask : 00100110
        -----------
        children    : aabaabba, bbabbaab

        :param parent1: One of the individuals from which generate the progeny.
        :param parent2: The other.
        :return: A list of two individuals, each a child containing some
            characteristics from their parents.
        """
        child1, child2 = super().__call__(parent1, parent2)

        for i in range(len(parent1)):
            if take_chances(.5):
                child1[i], child2[i] = parent2[i], parent1[i]
        return child1, child2


class SwapGenes(Mutation):
    """ Mutates the by swapping two random genes.

    This mutation method operates only with ListIndividuals (or any of their
    subclasses.
    """

    def __call__(self, individual, p):
        """ Swaps the values of two positions of the list of values.

        When the individual is mutated, two random positions (pivots) are
        generated. Then, the values of those positions are swapped. For example:

        individual : 12345678
        pivot      : 3, 5
        -----------
        mutated    : 12365478

        :param individual: The individual to be mutated.
        :param p: The probability of mutation.
        :return: A new individual mutated with a probabiity of p or looking
            exactly to the one passed as parameter with a probability of 1-p.
        """
        clone = individual.clone()
        if take_chances(probability=p):
            # Get two random diferent indexes
            indexes = range(len(individual))
            i1, i2 = tuple(random.sample(indexes, 2))
            # Swap the genes in the cloned individual
            clone[i1], clone[i2] = clone[i2], clone[i1]
            return clone
        else:
            return clone


class SingleGeneRandomValue(Mutation):
    """ Mutates the individual by changing the value to a random gene. """

    def __init__(self, alleles):
        """ Initializes this object.

        :param alleles: The set of values to choose from.
        """
        super().__init__()
        self.alleles = alleles

    def __call__(self, individual, p):
        """ Changes the value of a random gene of the individual.

        The mutated chromosome is obtained by changing a random gene as seen in
        the next example:

        individual : aabbaaba
        alleles    : (a, b, c, d)
        change pos : 7
        -----------
        mutated    : aabdaabc

        :param individual: The individual to be mutated.
        :param p: The probability of mutation.
        """
        clone = individual.clone()
        if take_chances(probability=p):
            # Set in a random position a different gene than before
            i = random.choice(range(len(individual)))
            new_gene = self.alleles.get()
            while individual[i] == new_gene:
                new_gene = self.alleles.get()
            # Set this gene in the cloned individual
            clone[i] = new_gene
            return clone
        else:
            return clone


class NGeneRandomValue(Mutation):
    """ Mutates the individual by changing the value to a random gene. """

    def __init__(self, alleles, n=None):
        """ Initializes this object.

        :param alleles: The set of values to choose from.
        :param n: The number of how many random genes may mutate. If greater
            than the size of individuals or None, the prone genes are all.
            Defaults to None.
        """
        super().__init__()
        self.alleles = alleles
        self.n = n

    def __call__(self, individual, p):
        """ Changes the value of a random gene of the individual.

        The mutated chromosome is obtained by changing a random gene as seen in
        the next example:

        individual : aabbaaba
        alleles    : (a, b, c, d)
        change pos : 7
        -----------
        mutated    : aabdaabc

        :param individual: The individual to be mutated.
        :param p: The probability of mutation.
        """
        indexes = random.sample(
            range(len(individual)),
            min(self.n or len(individual), len(individual)),
        )
        clone = individual.clone()
        for i in indexes:
            if take_chances(probability=p):
                new_gene = self.alleles.get()
                while individual[i] == new_gene:
                    new_gene = self.alleles.get()
                clone[i] = new_gene
        return clone
