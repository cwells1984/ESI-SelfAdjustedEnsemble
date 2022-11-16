from deap import base, creator, tools
import random
import numpy as np

# Global settings
MIN_VOTE = 1
MAX_VOTE = 10
NUM_CLASSIFIERS = 5


def generate_attribute():
    return np.random.randint(1,100)


# Fitness function
def evaluate(individual):

    # Create list tree_output = the predictions of the members of the ensemble
    tree_output = []
    for i in range(len(ENSEMBLE_OUTPUT[0])):
        res = (individual[0] * ENSEMBLE_OUTPUT[0][i]) + (individual[1] * ENSEMBLE_OUTPUT[1][i]) + \
                       (individual[2] * ENSEMBLE_OUTPUT[2][i]) + (individual[3] * ENSEMBLE_OUTPUT[3][i]) + \
                       (individual[4] * ENSEMBLE_OUTPUT[4][i])
        if res is None or type(res) is float:
            return 0.0,
        tree_output.append(res)

    # Compute tree's accuracy by comparing tree_output and y_expected
    acc = 0.0
    for i in range(len(tree_output)):
        c1 = np.argmax(tree_output[i])
        c2 = np.argmax(Y_EXPECTED[i])
        if c1 == c2:
            acc += 1.0
    return acc / len(tree_output),


# DEAP Setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attribute", generate_attribute)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=NUM_CLASSIFIERS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Setup operators
#toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


def evaluate_individual_fitnesses(_toolbox, _pop):
    fitnesses = map(_toolbox.evaluate, _pop)
    for ind, fit in zip(_pop, fitnesses):
        ind.fitness.values = fit


# The main execution of the evolutionary algorithm
def run(num_classifiers, pop_size, n_gen, ensemble_output, y_expected, verbose=True):

    global ENSEMBLE_OUTPUT
    ENSEMBLE_OUTPUT = ensemble_output
    global Y_EXPECTED
    Y_EXPECTED = y_expected

    # Now execute, first randomly generating the population and evaluating fitness
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(20)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    log = tools.Logbook()
    evaluate_individual_fitnesses(toolbox, pop)

    # Main loop
    for g in range(n_gen):

        # Select the next generation and clone
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

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
        evaluate_individual_fitnesses(toolbox, invalid_ind)

        # Finally replace the population with the offspring and update the hof
        pop[:] = offspring
        hof.update(pop)

        # Stream the statistics for this generation
        if verbose:
            record = stats.compile(pop)
            log.append(record)
            print(log.stream)

    return hof
