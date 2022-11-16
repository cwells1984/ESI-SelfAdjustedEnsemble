from deap import algorithms, base, creator, gp, tools
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from utilities import accuracy, lr_ensemble, data_prep, preprocess
import numpy as np

# GLOBAL SETTINGS HERE
NUM_CLASSIFIERS = 5
FRACTION=0.2
BASE_CLASSIFIER = DecisionTreeClassifier()
N_POP = 100
N_GEN = 50
PROB_CX = 0.5
PROB_MUT = 0.2


# PRIMITIVES
# If the value is not a list, return None
def listify(a):
    if type(a) is list:
        return a
    else:   # a is constant
        return None


def constify(c):
    if type(c) is float:
        return c
    else:   # c is list
        return None


# Add each element of lists a and b
def add_(a,b):
    a = listify(a)
    b = listify(b)
    if a is None or b is None:
        return None
    s = []
    for i in range(len(a)):
        s.append(a[i]+b[i])
    return s


# Subtract each element of lists a and b
def sub_(a,b):
    a = listify(a)
    b = listify(b)
    if a is None or b is None:
        return None
    s = []
    for i in range(len(a)):
        s.append(a[i]-b[i])
    return s


# Multiply each element of list a by a constant c
def mul_(a,c):
    a = listify(a)
    c = constify(c)
    if a is None or c is None:
        return None
    s = []
    for e in a:
        s.append(e*c)
    return s
# END PRIMITIVES

# DEAP SETUP
# The individual is an expression tree and we are attempting to maximize fitness (accuracy)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Create our primitive set
# +, -, x, and constant values 0.0 - 0.9
pset = gp.PrimitiveSet("MAIN", NUM_CLASSIFIERS)
pset.addPrimitive(add_, 2)
pset.addPrimitive(sub_, 2)
pset.addPrimitive(mul_, 2)
pset.addTerminal(0.0)
pset.addTerminal(0.1)
pset.addTerminal(0.2)
pset.addTerminal(0.3)
pset.addTerminal(0.4)
pset.addTerminal(0.5)
pset.addTerminal(0.6)
pset.addTerminal(0.7)
pset.addTerminal(0.8)
pset.addTerminal(0.9)

# Setup toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Register evolutionary operators
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0, indpb=1.0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pop = toolbox.population(n=1)
    for p in pop:
        print(p)
        toolbox.mutate(p)
        print(p)
