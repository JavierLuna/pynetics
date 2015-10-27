from pynetics import FitnessMethod, GeneticAlgorithm, Population
from pynetics.catastrophe import NoCatastrophe
from pynetics.ga_list import TwoPointCrossover, RandomGeneValue
from pynetics.ga_list.ga_bin import BinaryIndividualSpawningPool, binary_alleles
from pynetics.replacement import LowElitism
from pynetics.selection import BestIndividual
from pynetics.stop import StepsNumStopCondition

population_size = 100
replacement_rate = 100
individual_size = 30


class MaximizeOnesFitness(FitnessMethod):
    """ Fitness where more 1's implies higher fitness. """

    def perform(self, individual):
        return sum(individual)


if __name__ == '__main__':

    ga = GeneticAlgorithm(
        StepsNumStopCondition(100),
        [
            Population(
                population_size,
                replacement_rate,
                BinaryIndividualSpawningPool(individual_size),
                MaximizeOnesFitness()
            )
        ],
        BestIndividual(),
        LowElitism(),
        TwoPointCrossover(),
        RandomGeneValue(binary_alleles),
        NoCatastrophe(),
        0.85,
        0.01,
    )

    ga.run()
    for individual in ga.populations[0]:
        print(individual)
