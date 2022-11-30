import math

from deap import base, creator, tools
import random
import numpy as np

# Global settings
MIN_VOTE = 0.01
MAX_VOTE = 1.0


def generate_attribute():
    return np.random.random()


class Evolutionary:

    def __init__(self, num_classifiers):
        self.num_classifiers = num_classifiers

        # DEAP Setup
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax, old_fitness=0.0)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", generate_attribute)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attribute, n=self.num_classifiers*2)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Setup operators
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate)

    # Fitness function
    def evaluate(self, individual):

        # Create list tree_output = the predictions of the members of the ensemble
        tree_output = []
        for i in range(len(self.ensemble_output[0])):
            res = np.array([0.0, 0.0])
            for j in range(self.num_classifiers):
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

    # Perform log-normal self-adaptation
    def self_adapt_log(self, sigma, fitness_delta):

        theta = 0
        if fitness_delta > 0:
            theta = 0.5
        if fitness_delta < 0:
            theta = -1

        tau_global = 1 / math.sqrt(2 * self.num_classifiers)
        tau_local = 1 / math.sqrt(2 * math.sqrt(self.num_classifiers))
        e_exp = theta * np.abs((tau_global * np.random.normal(0.0, 1.0)) + (tau_local * np.random.normal(0.0, 1.0)))
        new_sigma = sigma * math.e ** e_exp
        return new_sigma

    # Mutate the individual
    def mutate_individual(self, _individual):
        delete_fitness = False
        for i in range(self.num_classifiers):

            # update this attribute's sigma value
            fitness_delta = _individual.fitness.values[0] - _individual.old_fitness
            sigma = _individual[i + self.num_classifiers]
            _individual[i + self.num_classifiers] = self.self_adapt_log(sigma, fitness_delta)

            # now mutate the attribute
            new_attribute = _individual[i] + np.random.normal(0, _individual[i + self.num_classifiers])
            if new_attribute < MIN_VOTE:
                new_attribute = MIN_VOTE
            if new_attribute > MAX_VOTE:
                new_attribute = MAX_VOTE
            _individual[i] = new_attribute
            delete_fitness = True

        if delete_fitness:
            _individual.old_fitness = _individual.fitness.values[0]
            del _individual.fitness.values

        return delete_fitness

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
                mutated = self.mutate_individual(mutant)

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
