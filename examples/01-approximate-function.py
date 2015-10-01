import math
from pynetics.algorithm import GeneticAlgorithmOld
from pynetics.allele import RangeAlleles
from pynetics.catastrophe import DoomsdayCatastrophe
from pynetics.crossover import MorphologicalCrossover
from pynetics.individual import ListIndividual
from pynetics.mutator import RandomGeneAlphabetListMutator
from pynetics.replacement import HighElitistReplacement
from pynetics.selector import TournamentSelector
from pynetics.stop_criteria import IterationStopCriteria


class ApproximateFunction(ListIndividual):
    """ Individual with real chromosome of the form [.34, .98, .001, ...]. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__f = kwargs['f']

    def fitness(self):
        """ Fitness is the RMSE of the function in values [0.0, 0.1, ..., 10.0] and the individual polynomial.

        :returns: The RMSE
        """
        sum_error = 0.0
        steps = 100
        for i in range(steps):
            x1 = self.__f(i / 10.)
            x2 = self.phenotype()(i / 10.)
            sum_error += pow(x2 - x1, 2)
        return sum_error / steps

    def phenotype(self):
        def f(x):
            y = 0
            for i in range(len(self.chromosome)):
                y += self.chromosome[i] * pow(x, i)
            return y

        return f

    def __str__(self):
        # TODO TBD
        return '{}'.format(', '.join([str(i) for i in self.chromosome]))


def f1(x):
    return 0.5 * x ** 2 + 0.3 * x + 0.9


def f2(x):
    return abs(math.sin(x))


if __name__ == '__main__':
    individual = ApproximateFunction(f=f1, size=3, alleles=RangeAlleles(0.0, 1.0))

    ga = GeneticAlgorithmOld(
        individual=individual,
        population_size=50,
        f_selection=TournamentSelector(5),
        f_crossover=MorphologicalCrossover(),
        f_mutation=RandomGeneAlphabetListMutator(),
        f_replacement=HighElitistReplacement(),
        f_stop_criteria=IterationStopCriteria(100000),
        p_crossover=0.9,
        p_mutation=0.1,
        f_catastrophe=DoomsdayCatastrophe(),
        p_catastrophe=0.01,
    )
    ga.evolve()
    print(ga.population[0])
