# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import random
from deap import base
from deap import creator
from deap import tools

# DEAP SETUP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator, creates toolbox.attr_bool() function returning a boolean
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers, individual() returns a 100-item long individual, population(n) returns a list of n individuals
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# The fitness function to be maximized
def eval_one_max(individual):
    return sum(individual)

# Register genetic operators
toolbox.register("evaluate", eval_one_max)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# END DEAP SETUP

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Create the population
    pop = toolbox.population(n=300)

    # Set probability of crossover and mutation
    CXPB = 0.5
    MUTPB = 0.2

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)
    print(f"Evaluated {len(pop)} individuals")

    # Extract all fitnesses
    fits = []
    for ind in pop:
        fits += ind.fitness.values

    # var for # of generations
    g = 0

    # Find the best individual
    best_ind = tools.selBest(pop, 1)[0]
    print(f"Best individual is {best_ind}, {best_ind.fitness.values}")

    # Begin the evolution
    while max(fits) < 100 and g < 1000:
        g += 1
        print(f"-- Generation {g} --")

        # Select the next generation of individuals and clone them
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover to the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # remove fitness values of the children - they must be recalculated
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation to the offspring with probability MUTPB and remove fitness
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Find individuals whose fitness needs updating
        invalid_ind = []
        for ind in offspring:
            if not ind.fitness.valid:
                invalid_ind += [ind]

        # Re-evaluate fitnesses
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)
        print(f"Re-evaluated {len(invalid_ind)} individuals")

        # Replace the population with the offspring
        pop[:] = offspring

        # Extract all fitnesses and get the stats
        fits = []
        for ind in pop:
            fits += ind.fitness.values
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(f ** 2 for f in fits)
        std = abs(sum2 / length - mean**2)**0.5

        # Print the stats
        print(f"  Min {min(fits)}")
        print(f"  Max {max(fits)}")
        print(f"  Avg {mean}")
        print(f"  Std {std}")

    print("-- End of (successful) evolution --")

    # Find the best individual
    best_ind = tools.selBest(pop, 1)[0]
    print(f"Best individual is {best_ind}, {best_ind.fitness.values}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
