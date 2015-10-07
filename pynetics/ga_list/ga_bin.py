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
    parents_num = 2

    def perform(self, individuals):
        """ Applies the crossover operator.

        :param individuals: The individuals to cross to generate progeny.
        :return: A list of individuals.
        """
        i1, i2 = individuals[0], individuals[1]
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
        return [child1, child2, ]
