import random

from pynetics import ga_list
from pynetics.ga_list import ListIndividualSpawningPool, \
    FixedLengthListRecombination


class BinaryIndividualSpawningPool(ListIndividualSpawningPool):
    """ Defines the methods for creating binary individuals. """

    def __init__(self, size, fitness):
        """ Initializes this spawning pool for generating list individuals.

        :param size: The size of the individuals to be created from this
            spawning pool.
        :param fitness: The fitness to use to evaluate the individuals generated
            by this SpawningPool instance.
        """
        super().__init__(size, binary_alleles, fitness)


class GeneralizedRecombination(FixedLengthListRecombination):
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
        bin_c = [int(x) for x in bin(c)[2:]]
        bin_d = [int(x) for x in bin(d)[2:]]

        # Convert to chromosomes and we're finish
        for i in range(len(bin_c)):
            child1[i], child2[i] = bin_c[i], bin_d[i]
        return child1, child2


# A Finite set of alleles where the only valid values are 0 and 1.
binary_alleles = ga_list.FiniteSetAlleles((0, 1))