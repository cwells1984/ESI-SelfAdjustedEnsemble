import deap.gp
from deap import base, creator, gp, tools
import operator
import preprocess
import random

NUM_FEATURES = 9


# DEAP SETUP

# The fitness function to evaluate
def evaluate(individual):
    func = toolbox.compile(expr=individual)
    sum_entries = 0
    for i in range(len(X)):
        sum_entries += func(X[i][0], X[i][1], X[i][2], X[i][3], X[i][4], X[i][5], X[i][6], X[i][7], X[i][8])
    avg = sum_entries / len(X)
    return avg,


# Individuals are primitive trees
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Create our primitive set
pset = gp.PrimitiveSet("MAIN", NUM_FEATURES)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)

# Setup toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Setup operators
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mutate", deap.gp.mutNodeReplacement, pset=0.08)
toolbox.register("evaluate", evaluate)

# END DEAP SETUP

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # pre-process the data and prepare np arrays
    df_breast = preprocess.breast_cancer_wisconsin("./datasets/breast-cancer-wisconsin.data")
    df_breast = df_breast[0:1]
    X = df_breast.loc[:, df_breast.columns != 'Malignant'].values
    y = df_breast.loc[:, df_breast.columns == 'Malignant'].values.ravel()

    # generate population and evaluate initial fitness
    pop = toolbox.population(n=100)
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Main loop, execute for a number of generations
    for g in range(1):
        # Select the next generation
        offspring = toolbox.select(pop, len(pop))
        # clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # apply mutation in offspring
        for mutant in offspring:
            print(mutant)
            #print(pset.terminals[mutant])
            if random.random() < 0.08:
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
