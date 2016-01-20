import sys

from pynetics import NoMutation, Population, Fitness
from pynetics.algorithms import GeneticAlgorithm
from pynetics.catastrophe import NoCatastrophe
from pynetics.ga_list import ListIndividualSpawningPool, FiniteSetAlleles
from pynetics.ga_list.ga_int import IntegerRangeRecombination, \
    IntegerIndividualSpawningPool
from pynetics.replacements import LowElitism
from pynetics.selections import BestIndividual, Tournament
from pynetics.stop import StepsNumStopCondition

NUM_INPUTS = 5
NUM_OUTPUTS = 1
MAX_NUM_OF_FUZZY_SETS_PER_LINGUISTIC_VAR = 10
DUMMY_OBJECTIVE_INDIVIDUAL = [3, 4, 2, 7, 5, 2]


class FuzzyControllerTopologyFitness(Fitness):
    """ Calcula el fitness de una topología dado el individuo que la representa.

    El individuo será de la clase IntListIndividual, representando cada gen el número de conjuntos difusos que tiene
    cada variable de entrada y salida. El número deberá estar comprendido entre 2 (el número mínimo de conjuntos difusos
    que puede tener una variable lingüística) y el límite superior (que puede ser especificado en el constructor). En
    caso de que alguno no sea válido, el fitness que se devolverá será directamente 0 ya que no es valido.
    """

    def __init__(self, max_num_of_fuzzy_sets=None):
        """ Construlle el objeto qe calcula el fitness.

        :param max_num_of_fuzzy_sets: El número máximo de conjuntos difusos que pueden tener las variables lingüísticas.
            Es opcional, y si no se espcifica tomará un valor de sys.max_int (vamos, que mucho).
        """
        self.max_num_of_fuzzy_sets = max_num_of_fuzzy_sets or sys.maxsize

    def perform(self, individual):
        if not any([2 <= x < self.max_num_of_fuzzy_sets for x in individual]):
            return 0
        else:
            # TODO Cuando esto esté, incluir la optimización de topología con coop-ga. De momento dummy.
            diffs = [abs(x - y) for x, y in zip(individual, DUMMY_OBJECTIVE_INDIVIDUAL)]
            if sum(diffs) == 0:
                return 2
            else:
                return 1.0 / sum(diffs)


def create_ga_for_fuzzy_controller(inputs, outputs):
    return GeneticAlgorithm(
        stop_condition=StepsNumStopCondition(100),
        populations=[
            Population(
                name='Fuzzy controllers',
                size=100,
                replacement_rate=0.9,
                spawning_pool=IntegerIndividualSpawningPool(
                    size=inputs + outputs,
                    lower=2,
                    upper=MAX_NUM_OF_FUZZY_SETS_PER_LINGUISTIC_VAR,
                ),
                fitness=FuzzyControllerTopologyFitness(MAX_NUM_OF_FUZZY_SETS_PER_LINGUISTIC_VAR),
                selection=Tournament(10),
                recombination=IntegerRangeRecombination(),
                mutation=NoMutation(),
                replacement=LowElitism(),
                p_recombination=1,
                p_mutation=0,
            )
        ],
        catastrophe=NoCatastrophe(),
    )

if __name__ == '__main__':
    ga = create_ga_for_fuzzy_controller(NUM_INPUTS, NUM_OUTPUTS)
    print('Objective: {}'.format(DUMMY_OBJECTIVE_INDIVIDUAL))
    print(ga.populations[0][0])
    ga.run()
    print(ga.populations[0][0])
