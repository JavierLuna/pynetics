from pynetics import individuals, fitnesses


class DummyIndividual(individuals.Individual):
    def phenotype(self):
        return None


class DummySpawningPool(individuals.SpawningPool):
    def create(self) -> DummyIndividual:
        return DummyIndividual()


class DummyFitness(fitnesses.Fitness):
    def perform(self, individual):
        return 0.5
