from deap import base, creator, tools
import random
import numpy as np

# Global settings
MIN_VOTE = 0.01
MAX_VOTE = 1.0
SIGMA = 0.1


def generate_attribute():
    return np.random.random()


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
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=SIGMA, indpb=0.1)
        self.toolbox.register("evaluate", self.evaluate)

    # Fitness function
    def evaluate(self, individual):

        # Create list tree_output = the predictions of the members of the ensemble
        tree_output = []
        for i in range(len(self.ensemble_output[0])):
            res = np.array([0.0, 0.0])
            for j in range(len(individual)):
                res += individual[j] * np.array(self.ensemble_output[j][i])
            res = res.tolist()
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

    # Mutate the individual
    def mutate_individual(self, _individual):
        delete_fitness = False
        for i in range(len(_individual)):
            new_attribute = _individual[i] + np.random.normal(0, SIGMA)
            if new_attribute < MIN_VOTE:
                new_attribute = MIN_VOTE
            if new_attribute > MAX_VOTE:
                new_attribute = MAX_VOTE
            _individual[i] = new_attribute
            delete_fitness = True

        if delete_fitness:
            del _individual.fitness.values

        return delete_fitness

    # The main execution of the evolutionary algorithm
    def run(self, pop_size, n_gen, ensemble_output, y_expected, verbose=True):
        self.ensemble_output = ensemble_output
        self.y_expected = y_expected

        # Now execute, first randomly generating the population and evaluating fitness
        pop = self.toolbox.population(n=pop_size)

        hof = tools.HallOfFame(1)
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
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

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
