from deap import base, creator, tools
import random
import numpy as np

# Global settings
MIN_VOTE = 1
MAX_VOTE = 10


def generate_attribute():
    return np.random.randint(1, 100)


class Evolutionary:

    def __init__(self, num_classifiers):

        # DEAP Setup
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", generate_attribute)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attribute, n=num_classifiers)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Setup operators
        # toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate)

    # Fitness function
    def evaluate(self, individual):

        # Create list tree_output = the predictions of the members of the ensemble
        tree_output = []
        for i in range(len(self.ensemble_output[0])):
            res = (individual[0] * self.ensemble_output[0][i]) + (individual[1] * self.ensemble_output[1][i]) + \
                           (individual[2] * self.ensemble_output[2][i]) + (individual[3] * self.ensemble_output[3][i]) + \
                           (individual[4] * self.ensemble_output[4][i])
            if res is None or type(res) is float:
                return 0.0,
            tree_output.append(res)

        # Compute tree's accuracy by comparing tree_output and y_expected
        acc = 0.0
        for i in range(len(tree_output)):
            c1 = np.argmax(tree_output[i])
            c2 = np.argmax(self.y_expected[i])
            if c1 == c2:
                acc += 1.0
        return acc / len(tree_output),

    # Evaluate the fitness values of the individuals in the population
    def evaluate_individual_fitness(self, _pop):
        fitness_values = map(self.toolbox.evaluate, _pop)
        for ind, fit in zip(_pop, fitness_values):
            ind.fitness.values = fit

    # The main execution of the evolutionary algorithm
    def run(self, pop_size, n_gen, ensemble_output, y_expected, verbose=True):
        self.ensemble_output = ensemble_output
        self.y_expected = y_expected

        # Now execute, first randomly generating the population and evaluating fitness
        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(20)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        log = tools.Logbook()
        self.evaluate_individual_fitness(pop)

        # Main loop
        for g in range(n_gen):

            # Select the next generation and clone
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # apply mutation in offspring
            for mutant in offspring:
                if random.random() < 0.1:
                    #toolbox.mutate(mutant)
                    mutant[0] += np.random.randint(-10, 10)
                    del mutant.fitness.values
                    for i in range(len(mutant)):
                        if mutant[i] < MIN_VOTE:
                            mutant[i] = MIN_VOTE
                        if mutant[i] > MAX_VOTE:
                            mutant[i] = MAX_VOTE

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self.evaluate_individual_fitness(invalid_ind)

            # Finally replace the population with the offspring and update the hof
            pop[:] = offspring
            hof.update(pop)

            # Stream the statistics for this generation
            if verbose:
                record = stats.compile(pop)
                log.append(record)
                print(log.stream)

        return hof
