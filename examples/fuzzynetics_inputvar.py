from pynetics import NoMutation, Population, Fitness
from pynetics.algorithms import GeneticAlgorithm
from pynetics.catastrophe import NoCatastrophe
from pynetics.ga_list import ListIndividualSpawningPool
from pynetics.ga_list.ga_real import RealIntervalAlleles, \
    MorphologicalRecombination
from pynetics.replacements import LowElitism
from pynetics.selections import Tournament
from pynetics.stop import StepsNumStopCondition

NUM_INPUTS = 5
NUM_OUTPUTS = 1
MAX_NUM_OF_FUZZY_SETS_PER_LINGUISTIC_VAR = 10
DUMMY_OBJECTIVE_INDIVIDUAL = [3, 4, 2, 7, 5, 2]


class ControllerFitness(Fitness):
    """ Calcula el fitness de un controlador en concreto. """

    def perform(self, individual):
        pass


def input_var_population(name, domain, num_fs):
    return Population(
            name=name,
            size=100,
            replacement_rate=0.9,
            spawning_pool=ListIndividualSpawningPool(
                    size=2 * (num_fs - 1),
                    alleles=RealIntervalAlleles(domain[0], domain[1])
            ),
            fitness=ControllerFitness(),
            selection=Tournament(10),
            recombination=MorphologicalRecombination(),
            mutation=NoMutation(),
            replacement=LowElitism(),
            p_recombination=0.9,
            p_mutation=0,
    )


def output_var_population(name, domain, num_fs):
    return Population(
            name=name,
            size=100,
            replacement_rate=0.9,
            spawning_pool=ListIndividualSpawningPool(
                    size=num_fs,
                    alleles=RealIntervalAlleles(domain[0], domain[1])
            ),
            fitness=ControllerFitness(),
            selection=Tournament(10),
            recombination=MorphologicalRecombination(),
            mutation=NoMutation(),
            replacement=LowElitism(),
            p_recombination=0.9,
            p_mutation=0,
    )


def create_ga_for_fuzzy_controller(inputs, outputs):
    return GeneticAlgorithm(
            stop_condition=StepsNumStopCondition(100),
            populations=[
                input_var_population(),
            ],
            catastrophe=NoCatastrophe(),
    )


if __name__ == '__main__':
    ga = create_ga_for_fuzzy_controller(NUM_INPUTS, NUM_OUTPUTS)
    print('Objective: {}'.format(DUMMY_OBJECTIVE_INDIVIDUAL))
    print(ga.populations[0][0])
    ga.run()
    print(ga.populations[0][0])
