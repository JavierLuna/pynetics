import random
from pynetics import ga_list


class IntegerIndividualSpawningPool(ga_list.ListIndividualSpawningPool):
    """ Defines the methods for creating integer individuals. """

    def __init__(self, size, lower, upper):
        """ Initializes this spawning pool for generating list individuals.

        :param size: The size of the individuals to be created from this
            spawning pool.
        :param lower: the lower bound of the integer set (included).
        :param upper: the upper bound of the integer set (included).
        """
        super().__init__(
            size,
            ga_list.FiniteSetAlleles(range(lower, upper + 1))
        )
        self.lower = lower
        self.upper = upper


class IntegerRangeRecombination(ga_list.ListRecombination):
    """ Offspring is obtained by crossing individuals gene by gene.

    For each gene, the interval of their values is calculated. Then, the
    difference of the interval is used for calculating the new interval from
    where to pick the values of the new genes. First, a value is taken from the
    new interval. Second, the other value is calculated by taking the
    symmetrical by the center of the range.
    """
    def perform(self, parent1, parent2):
        """ Applies the crossover operator.

        :param parent1: One of the individuals from which generate the progeny.
        :param parent2: The other.
        :return: A list of individuals.
        """
        i1_lower = parent1.population.spawning_pool.lower
        i1_upper = parent1.population.spawning_pool.upper
        i2_lower = parent2.population.spawning_pool.lower
        i2_upper = parent2.population.spawning_pool.upper
        child1, child2 = parent1.population.spawn(), parent2.population.spawn()
        for i, genes in enumerate(zip(parent1, parent2)):
            # For each gene, we calculate the the crossover interval. If the
            # genes are equal, we take the whole possible interval
            a, b = genes
            if a != b:
                diff = abs(a - b)
            else:
                diff = abs(min(i1_lower, i2_lower) - max(i1_upper, i2_upper))

            child1[i] = max(
                min(random.randint(a - diff, b + diff), i1_upper),
                i1_lower
            )
            child2[i] = max(
                min(a + b - child1[i], i2_upper),
                i2_lower
            )  # Just in case individuals generated by different spawning pools.
        return [child1, child2, ]