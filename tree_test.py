# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import random
from deap import base
from deap import creator
from deap import gp
from deap import tools

# Number of classifiers
num_classifiers = 4

# PRIMITIVES

# If the value is not a list, return None
def listify(a):
    if type(a) is list:
        return a
    else:   # a is constant
        return None

# If the value is not a float, return None
def constantify(c):
    if type(c) is float:
        return c
    else:
        return None

# Add the list elements of a and b (if not lists return None)
def add_(a,b):
    return a + b

# Muiltiplies every element of a list a by some constant c (returns None if types arent correct)
def mult_(a,c):
    a = listify(a)
    c = constantify(c)
    if a is None or c is None:
        return None
    s = []
    for i in a:
        s += [i * c]

# END PRIMITIVES

# DEAP SETUP

# the individual is now a tree
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Create our primitive set
pset = gp.PrimitiveSet("MAIN", num_classifiers)
pset.addPrimitive(add_, 2)
#pset.addPrimitive(mult_, 2)
#pset.addTerminal(0.0)
#pset.addTerminal(0.1)
#pset.addTerminal(0.2)
#pset.addTerminal(0.3)
#pset.addTerminal(0.4)
#pset.addTerminal(0.5)
#pset.addTerminal(0.6)
#pset.addTerminal(0.7)
#pset.addTerminal(0.8)
#pset.addTerminal(0.9)

# Setup toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# The fitness function to be maximized
def eval_mod(individual):

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    ens_output = []
    ens_output = func(models_output[0], models_output[1], models_output[2], models_output[3])
    print(ens_output)

    # Compute accuracy
    acc = 0.0
    for i in range(ens_output):
        c1 = numpy.argmax


toolbox.register("evaluate", eval_mod)

# END DEAP SETUP

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # dummy base classifier results
    models_output = np.zeros((4, 1))
    models_output[0,:] = 1
    models_output[1,:] = 0
    models_output[2,:] = 0
    models_output[3,:] = 0

    # generate population
    pop = toolbox.population(n=1)

    print("CREATED TREES")
    for ind in pop:
        print(ind)

    print("EVALUATED TREES")
    fitnesses = list(map(toolbox.evaluate, pop))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
