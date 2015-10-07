from pynetics.base import GeneticAlgorithmOld
from pynetics.allele import BinaryAlleles
from pynetics.catastrophe import DoomsdayCatastrophe
from pynetics.crossover import GeneralizedCrossover
from pynetics.individual import ListIndividual
from pynetics.mutator import RandomGeneAlphabetListMutator
from pynetics.replacement import HighElitistReplacement
from pynetics.selector import TournamentSelector
from pynetics.stop_criteria import IterationStopCriteria


class MaximizeOnesIndividual(ListIndividual):
    """ List chromosome of the form '0100111001'. """

    def phenotype(self):
        return sum(self.chromosome)

    def fitness(self):
        """ Fitness is proportional to the number of 1s in chromosome.

        :returns: The number of ones the chromosome has.
        """
        return self.phenotype() / len(self.chromosome)

    def __str__(self):
        # TODO TBD
        return '{} -> {}'.format(''.join([str(i) for i in self.chromosome]), self.fitness())


if __name__ == '__main__':
    individual = MaximizeOnesIndividual(size=20, alleles=BinaryAlleles())

    ga = GeneticAlgorithmOld(
        individual=individual,
        population_size=20,
        f_selection=TournamentSelector(10),
        f_crossover=GeneralizedCrossover(),
        f_mutation=RandomGeneAlphabetListMutator(),
        f_replacement=HighElitistReplacement(),
        f_stop_criteria=IterationStopCriteria(10000),
        p_crossover=0.9,
        p_mutation=0.1,
        maximize_fitness=True,
        f_catastrophe=DoomsdayCatastrophe(),
        p_catastrophe=0.05,
    )
    ga.step()
    print(ga.population[0])
