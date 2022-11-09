import deap.gp
from deap import algorithms, base, creator, gp, tools
import operator
import preprocess
import random
from sklearn.model_selection import train_test_split
import numpy as np


# DEAP SETUP

# Custom Primitives are here
# protected div
def div(a, b):
    if b == 0:
        return 0
    else:
        return a / b


# maximum
def max(a, b):
    if a >= b:
        return a
    else:
        return b


# minimum
def min(a, b):
    if a <= b:
        return a
    else:
        return b


# Calculates the accuracy of the ensemble using majority voting
def calc_ensemble_predictions(ensemble, _X, _y):
    _y_pred_votes = np.zeros((len(_y), 2))

    for individual in ensemble:
        y_pred_individual = calc_tree_predictions(individual, _X, _y)
        for i in range(len(y_pred_individual)):
            _y_pred_votes[i][0] += y_pred_individual[i][0]
            _y_pred_votes[i][1] += y_pred_individual[i][1]

    #print(_y_pred_votes)
    return _y_pred_votes


def calc_ensemble_accuracy(ensemble, _X, _y):
    y_pred = np.argmax(calc_ensemble_predictions(ensemble, _X, _y), axis=1)
    accuracy = 0

    # Calculate accuracy
    for i in range(len(y_pred)):
        if y_train[i] == y_pred[i]:
            accuracy += 1
    accuracy = accuracy / len(y_pred)

    return accuracy,


# Helper method - compiles, evaluates, and scores accuracy of a tree with a given data set
def calc_tree_predictions(individual, _X, _y):
    func = toolbox.compile(expr=individual)
    _y_pred = np.zeros((len(_y), 2))

    # Get tree output - if >= 0 True, if not False
    for i in range(len(_X)):
        tree_out = func(_X[i][0], _X[i][1], _X[i][2], _X[i][3], _X[i][4], _X[i][5], _X[i][6], _X[i][7], _X[i][8])
        if tree_out >= 0:
            _y_pred[i][1] += 1
        else:
            _y_pred[i][0] += 1

    return _y_pred


def calc_tree_accuracy(individual, _X, _y):
    y_pred = np.argmax(calc_tree_predictions(individual, _X, _y), axis=1)
    accuracy = 0

    # Calculate accuracy
    for i in range(len(y_pred)):
        if y_train[i] == y_pred[i]:
            accuracy += 1
    accuracy = accuracy / len(y_pred)

    return accuracy,


# The fitness function to evaluate
def evaluate(individual):
    return calc_tree_accuracy(individual, X_train, y_train)


# Individuals are primitive trees
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Create our primitive set
pset = gp.PrimitiveSet("MAIN", 9)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)

# Initialization
toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Set operators
toolbox.register("select", tools.selTournament, tournsize=3)
#toolbox.register("select", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", deap.gp.mutNodeReplacement, pset=pset)
toolbox.register("evaluate", evaluate)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # pre-process the data and prepare np arrays
    df_breast = preprocess.breast_cancer_wisconsin("./datasets/breast-cancer-wisconsin.data")
    X = df_breast.loc[:, df_breast.columns != 'Malignant'].values
    y = df_breast.loc[:, df_breast.columns == 'Malignant'].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # now execute the EA
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(20)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)

    # Now try the top performer on the test set
    top_ind = hof[0]
    accuracy = calc_tree_accuracy(top_ind, X_test, y_test)
    print(f"Accuracy of Top Performer= {accuracy[0]*100:.3f}%")

    accuracy = calc_ensemble_accuracy(hof, X_test, y_test)
    print(f"Accuracy of Ensemble= {accuracy[0] * 100:.3f}%")
