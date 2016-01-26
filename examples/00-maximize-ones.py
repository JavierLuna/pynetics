from pynetics import Fitness, Population
from pynetics.algorithms import GeneticAlgorithm
from pynetics.catastrophe import NoCatastrophe
from pynetics.ga_list import TwoPointRecombination, RandomGeneValue
from pynetics.ga_list.ga_bin import BinaryIndividualSpawningPool, binary_alleles
from pynetics.replacements import LowElitism
from pynetics.selections import BestIndividual
from pynetics.stop import StepsNumStopCondition

population_size = 100
replacement_rate = 100
individual_size = 30


class MaximizeOnesFitness(Fitness):
    """ Fitness where more 1's implies higher fitness. """

    def perform(self, individual):
        return sum(individual)


if __name__ == '__main__':
    ga = GeneticAlgorithm(
        StepsNumStopCondition(100),
        [
            Population(
                name='1',
                size=population_size,
                replacement_rate=0.8,
                spawning_pool=BinaryIndividualSpawningPool(individual_size),
                fitness=MaximizeOnesFitness(),
                selection=BestIndividual(),
                recombination=TwoPointRecombination(),
                p_recombination=0.8,
                mutation=RandomGeneValue(binary_alleles),
                p_mutation=0.1,
                replacement=LowElitism(),
            )
        ],
        NoCatastrophe(),
    )

    print(ga.populations[0][0])
    ga.run()
    print(ga.populations[0][0])
