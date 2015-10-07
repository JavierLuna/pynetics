import abc
import random

from pynetics import SpawningPool, Individual, MutateMethod
from pynetics.crossover import CrossoverMethod
from pynetics.utils import take_chances


class Alleles(metaclass=abc.ABCMeta):
    """ The alleles are all the possible values a gene can take. """

    @abc.abstractmethod
    def get(self):
        """ Returns a random value of all the possible existent values. """


class FiniteSetAlleles(Alleles):
    """ The possible alleles belong to a finite set of symbols. """

    def __init__(self, values):
        """ Initializes this set of alleles with its sequence of symbols.

        :param values: The sequence of symbols.
        """
        self.__values = list(set(values))

    def get(self):
        """ A random value is selected uniformly over the set of values. """
        return random.choice(self.__values)


class ListIndividualSpawningPool(SpawningPool, metaclass=abc.ABCMeta):
    """ Defines the methods for creating individuals required by population. """

    def __init__(self, size, alleles):
        """ Initializes this spawning pool for generating list individuals.

        :param size: The size of the individuals to be created from this
            spawning pool.
        :param alleles: The alleles to be used as values of the genes.
        """
        self.__size = size
        self.__alleles = alleles

    def create(self):
        """ Creates a new individual randomly.

        :return: A new Individual object.
        """
        individual = ListIndividual()
        for _ in range(self.__size):
            individual.append(self.__alleles.get())
        return individual


class ListIndividual(list, Individual):
    """ An individual whose representation is a list of finite values. """

    def __eq__(self, individual):
        """ The equality between two list individuals is True if they:

        1. Have the same length
        2. Any two genes in the same position have the same value.
        """
        return len(self) == len(individual) and all(
            [x == y for (x, y) in zip(self, individual)]
        )


class ListCrossover(CrossoverMethod, metaclass=abc.ABCMeta):
    """ Common behavior for crossover methods for ListIndividual instances. """

    def __call__(self, individuals):
        """ Performs some checks before applying the crossover method.

        Specifically, it checks if the length of all individuals are the same.
        In so, the crossover operation is done. If not, a ValueError is raised.

        :param individuals: The individuals to cross to generate progeny.
        :return: A list of individuals with characteristics of the parents.
        :raises ValueError: If not all the individuals has the same length.
        """
        lengths = [len(i) for i in individuals]
        if not lengths.count(lengths[0]) == len(lengths):
            raise ValueError('Both individuals must have the same length')
        else:
            return super().__call__(individuals)


class OnePointCrossover(ListCrossover):
    """ Offspring is created by mixing the parents using one random pivot point.

    This crossover implementation works with two (and only two) individuals of
    type ListIndividual (or subclasses).
    """
    parents_num = 2

    def perform(self, individuals):
        """ Offspring is obtained mixing the parents with one pivot point.

        One example:

        parents  : aaaaaaaa, bbbbbbbb
        pivot    : 3
        -----------
        children : aaabbbbb, bbbaaaaa

        :param individuals: The individuals to cross to generate progeny.
        :return: A list of two individuals, each a child containing some
            characteristics from their parents.
        """
        i1, i2 = individuals[0], individuals[1]
        child1, child2 = i1.population.spawn(), i1.population.spawn()

        p = random.randint(1, len(i1) - 1)
        for i in range(len(i1)):
            child1[i], child2[i] = (i1[i], i2[i]) if i < p else (i2[i], i1[i])
        return [child1, child2, ]


class TwoPointCrossover(ListCrossover):
    """ Offspring is created by mixing the parents using two random pivot point.

    This crossover implementation works with two (and only two) individuals of
    type ListIndividual (or subclasses).
    """
    parents_num = 2

    def perform(self, individuals):
        """ Offspring is obtained mixing the parents with two pivot point.

        One example:

        parents  : aaaaaaaa, bbbbbbbb
        pivot    : 3, 5
        -----------
        children : aaabbaaa, bbbaabbb

        :param individuals: The individuals to cross to generate progeny.
        :return: A list of two individuals, each a child containing some
            characteristics from their parents.
        """
        i1, i2 = individuals[0], individuals[1]
        child1, child2 = i1.population.spawn(), i1.population.spawn()

        pivots = random.sample(range(len(i1) - 1), 2)
        p, q = min(pivots[0], pivots[1]), max(pivots[0], pivots[1])
        for i in range(len(i1)):
            child1[i], child2[i] = (i1[i], i2[i]) if p < i < q else (
                i2[i], i1[i])
        return [child1, child2, ]


class RandomMaskCrossover(ListCrossover):
    """ Offspring is created by using a random mask.

    This crossover implementation works with two (and only two) individuals of
    type ListIndividual (or subclasses).
    """

    parents_num = 2

    def perform(self, individuals):
        """ Offspring is obtained generating a random mask.

        This mask determines which genes of each of the progenitors are used on
        each of the the genes. For example:

        parents     : aaaaaaaa, bbbbbbbb
        random mask : 00100110
        -----------
        children    : aabaabba, bbabbaab

        :param individuals: The individuals to cross to generate progeny.
        :return: A list of two individuals, each a child containing some
            characteristics from their parents.
        """
        i1, i2 = individuals[0], individuals[1]
        child1, child2 = i1.population.spawn(), i1.population.spawn()

        for i in range(len(i1)):
            if take_chances(.5):
                child1[i], child2[i] = i1[i], i2[i]
            else:
                child1[i], child2[i] = i2[i], i1[i]
        return [child1, child2, ]


class SwapGenes(MutateMethod):
    """ Mutates the by swapping two random genes.

    This mutation method operates only with ListIndividuals (or any of their
    subclasses.
    """

    def perform(self, individual):
        """ Swaps the values of two positions of the list of values.

        When the individual is mutated, two random positions (pivots) are
        generated. Then, the values of those positions are swapped. For example:

        individual : 12345678
        pivot      : 3, 5
        -----------
        mutated    : 12365478

        :param individual: The individual to be mutated.
        """
        new_individual = individual.population.spawn()
        genes = random.sample(range(len(new_individual) - 1), 2)
        g1, g2 = genes[0], genes[1]
        new_individual.chromosome[g1], new_individual.genes[g2] = \
            new_individual.genes[g2], new_individual.genes[g1]
        return new_individual


class RandomGeneValue(MutateMethod):
    """ Mutates the individual by changing the value to a random gene. """

    def perform(self, individual):
        """ Changes the value of a random gene of the individual.

        The mutated chromosome is obtained by changing a random gene as seen in
        the next example:

        individual : aabbaaba
        alleles    : (a, b, c, d)
        change pos : 7
        -----------
        mutated    : aabdaabc

        :param individual: The individual to be mutated.
        """
        new_individual = individual.population.spawn()
        i = random.choice(range(len(individual)))
        new_individual.chromosome[i] = new_individual.alleles.get()
        return new_individual
