# From tutorial
# https://deap.readthedocs.io/en/master/overview.html
from deap import base, creator, tools
import random

# Parameters
IND_SIZE = 10
N = 50
CXPB = 0.5
MUTPB = 0.2
NGEN = 40

# Fitness function
def evaluate(individual):
    return sum(individual),


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Types
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Initialization
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Set operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # SETUP COMPLETE

    # Step 1: Randomly initialize population
    pop = toolbox.population(n=N)

    # Step 2: Evaluate fitness
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Main loop, execute for a number of generations
    for g in range(NGEN):
        # Select the next generation
        offspring = toolbox.select(pop, len(pop))
        # clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # apply crossover in offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # apply mutation in offspring
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        # for ind in offspring:
        #     print(ind)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Finally replace the population with the offspring
        pop[:] = offspring

        # calculate average fitness
        total_fitness = 0
        for ind in pop:
            total_fitness += ind.fitness.values[0]
        print(f"Average fitness: {total_fitness / len(pop)}")
