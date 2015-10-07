import random

from pynetics.ga_list import FiniteSetAlleles, ListCrossover


class BinaryAlleles(FiniteSetAlleles):
    """ Finite set of alleles which its possible values are only 0 and 1. """

    def __init__(self):
        """ Initializes this alleles with the two possible values. """
        super().__init__((0, 1))


class GeneralizedCrossover(ListCrossover):
    """ Offspring is obtained by crossing individuals as they where integers.

    NOTE: Works only for individuals with list chromosomes of binary alleles.
    Ok, may work for other individuals with list chromosomes, but the results
    may be strange (perhaps better, but I doubt it)
    """

    def perform(self, i1, i2):
        # Obtain the crossover range (as integer values)
        a = int(''.join([str(b1 & b2) for (b1, b2) in zip(i1, i2)]), 2)
        b = int(''.join([str(b1 | b2) for (b1, b2) in zip(i1, i2)]), 2)

        # Get the children (as integer values)
        c = random.randint(a, b)
        d = b - (c - a)

        # Convert to binary lists again (NOTE: we remove the starting 0b)
        bin_c = [int(x) for x in bin(c)[2:]]
        bin_d = [int(x) for x in bin(d)[2:]]

        # Convert to chromosomes and we're finish
        child1, child2 = i1.population.spawn(), i2.population.spawn()
        for i in range(len(bin_c)):
            child1[i], child2[i] = bin_c[i], bin_d[i]
        return child1, child2


class OnePointCrossover(ListCrossover):
    """ Offspring is created by mixing the parents using one random pivot point.

    This crossover implementation works with two (and only two) individuals of
    type ListIndividual (or subclasses).
    """

    def perform(self, population, i1, i2):
        """ Offspring is obtained mixing the parents with one pivot point.

        One example:

        parents  : aaaaaaaa, bbbbbbbb
        pivot    : 3
        -----------
        children : aaabbbbb, bbbaaaaa

        :param i1: The first parent.
        :param i2: The second parent.
        :return: A list of two individuals, each a child containing some
            characteristics from their parents.
        """
        child1, child2 = i1.population.spawn(), i2.population.spawn()

        p = random.randint(1, len(i1) - 1)
        for i in range(len(i1)):
            child1[i], child2[i] = (i1[i], i2[i]) if i < p else (i2[i], i1[i])
        return child1, child2
