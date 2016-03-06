from pynetics import StopCondition, Individual, SpawningPool, Fitness, Mutation, \
    Recombination, Replacement, Selection, Population


class DummyStopCondition(StopCondition):
    def __call__(self, genetic_algorithm):
        return True


class DummyIndividual(Individual):
    def phenotype(self):
        return 'DummyIndividual'

    def clone(self):
        individual = type(self)()
        individual.__dict__.update(self.__dict__)
        return individual


class DummySpawningPool(SpawningPool):
    def create(self):
        return DummyIndividual()


class DummyFitness(Fitness):
    def perform(self, individual):
        return 0.5


class DummyMutation(Mutation):
    def perform(self, individual):
        return individual.clone()


class DummyRecombination(Recombination):
    def perform(self, *args: Individual):
        return [i.clone() for i in args]


class DummyReplacement(Replacement):
    def perform(self, population, offspring):
        return population


class DummySelection(Selection):
    def perform(self, population, n):
        return population[:n]


class DummyCatastrophe(Selection):
    def perform(self, population, n):
        return population[:n]


class DummyPopulation(Population):
    def __init__(self, size, individuals=None):
        super().__init__(
            name='dummy',
            size=size,
            spawning_pool=DummySpawningPool(
                fitness=DummyFitness()
            ),
            replacement_rate=1.0,
            selection=DummySelection(),
            recombination=DummyRecombination(),
            p_recombination=1.0,
            mutation=DummyMutation(),
            p_mutation=0.1,
            replacement=DummyReplacement(),
            individuals=individuals,
        )
