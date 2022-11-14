import operator

import numpy as np
import preprocess
import task0_classifier_ensemble
from deap import algorithms, base, creator, gp, tools
from sklearn.model_selection import StratifiedKFold
import operator

NUM_CLASSIFIERS = 5


# PRIMITIVES
# If the value is not a list, return None
#
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


def add_(a,b):
    a = listify(a)
    b = listify(b)
    if a is None or b is None:
        return None
    s = []
    for i in range(len(a)):
        s.append(a[i]+b[i])
    return s


def sub_(a,b):
    a = listify(a)
    b = listify(b)
    if a is None or b is None:
        return None
    s = []
    for i in range(len(a)):
        s.append(a[i]-b[i])
    return s


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

# the individual is now a tree
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Create our primitive set
pset = gp.PrimitiveSet("MAIN", NUM_CLASSIFIERS)
pset.addPrimitive(add_, 2)
#pset.addPrimitive(sub_, 2)
pset.addPrimitive(mul_, 2)
# pset.addPrimitive(operator.add, 2)
# pset.addPrimitive(operator.sub, 2)
# pset.addPrimitive(operator.mul, 2)
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

# The fitness function to be maximized
def evaluate(individual):

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    tree_output = []
    for i in range(len(ensemble_output[0])):
        res = func(ensemble_output[0][i], ensemble_output[1][i], ensemble_output[2][i], ensemble_output[3][i], ensemble_output[4][i])
        if res is None or type(res) is float:
            return 0.0,
        tree_output.append(res)

    # Compute ensemble accuracy
    acc = 0.0
    for i in range(len(tree_output)):
        c1 = np.argmax(tree_output[i])
        c2 = np.argmax(y_test[i])
        if c1 == c2:
            acc += 1.0
    return acc / len(tree_output),

# def evaluate(individual):
#
#     # Transform the tree expression in a callable function
#     func = toolbox.compile(expr=individual)
#     tree_output = []
#     for i in range(len(ensemble_output[0])):
#         res = func(ensemble_output[0][i], ensemble_output[1][i], ensemble_output[2][i], ensemble_output[3][i],
#                    ensemble_output[4][i])
#         tree_output.append(res)
#
#     # Compute ensemble accuracy
#     acc = 0.0
#     for i in range(len(tree_output)):
#         if tree_output[i] == y_test[i]:
#             acc += 1.0
#     return acc / len(tree_output),


# Register evolutionary operators
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
toolbox.register("evaluate", evaluate)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # pre-process the data and prepare np arrays
    df_liver = preprocess.bupa_liver_disorders("./datasets/bupa.data")
    X = df_liver.loc[:, df_liver.columns != 'class'].values
    y = df_liver.loc[:, df_liver.columns == 'class'].values.ravel()

    # create the ensemble
    ensemble = task0_classifier_ensemble.create_ensemble(num_classifiers=NUM_CLASSIFIERS)

    # Using 5-fold cross validation calculate the accuracy of the ensemble
    accuracies = []
    ensemble_accuracies = []
    best_base_accuracies = []
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    fold_number = 1
    for train_index, test_index in kf.split(X, y):
        print(f"FOLD NUMBER {fold_number}")

        # train each item of the ensemble on a subset of the training data
        for clf in ensemble:
            task0_classifier_ensemble.train_classifier_on_subset(clf, X[train_index], y[train_index])

        # take the output of the ensembles' predictions on the training data, and train the aggregator via evolutionary runs
        accuracies = []
        ensemble_output = []
        for clf in ensemble:
            task0_classifier_ensemble.predict_classifier_on_subset(clf, X[train_index], y[train_index], ensemble_output,
                                                                   accuracies)

        # Onehot Encode the ensemble output
        ensemble_output_onehot = []
        for i in range(len(ensemble_output)):
            ensemble_output_onehot += [[]]
            for j in range(len(ensemble_output[i])):
                if ensemble_output[i][j] == 0:
                    ensemble_output_onehot[i] += [[1, 0]]
                else:
                    ensemble_output_onehot[i] += [[0, 1]]
        ensemble_output = ensemble_output_onehot

        # Using the output of the training data, evolve the aggregator
        pop = toolbox.population(n=100)
        hof = tools.HallOfFame(20)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        y_test = y[train_index]
        y_test_onehot = []
        for j in y_test:
            if j == 0:
                y_test_onehot += [[1, 0]]
            else:
                y_test_onehot += [[0, 1]]
        y_test = y_test_onehot
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)

        # make predictions for each base classifier
        accuracies = []
        ensemble_output = []
        for clf in ensemble:
            task0_classifier_ensemble.predict_classifier_on_subset(clf, X[test_index], y[test_index], ensemble_output, accuracies)

        # Onehot Encode the ensemble output
        ensemble_output_onehot = []
        for i in range(len(ensemble_output)):
            ensemble_output_onehot += [[]]
            for j in range(len(ensemble_output[i])):
                if ensemble_output[i][j] == 0:
                    ensemble_output_onehot[i] += [[1, 0]]
                else:
                    ensemble_output_onehot[i] += [[0, 1]]
        ensemble_output = ensemble_output_onehot

        # feed these estimates into our best aggregator and get the aggregator's accuracy
        y_test = y[test_index]
        y_test_onehot = []
        for j in y_test:
            if j == 0:
                y_test_onehot += [[1, 0]]
            else:
                y_test_onehot += [[0, 1]]
        y_test = y_test_onehot
        best_aggregator_accuracy = 0
        best_pop_aggregator = None
        best_pop_aggregator = hof[0]
        best_aggregator_accuracy = evaluate(hof[0])[0]
        # for pop_aggregator in pop:
        #     current_accuracy = evaluate(pop_aggregator)[0]
        #     if current_accuracy > best_aggregator_accuracy:
        #         best_aggregator_accuracy = current_accuracy
        #         best_pop_aggregator = pop_aggregator
        print(best_pop_aggregator)
        print(f"Evolved Aggregator accuracy= {best_aggregator_accuracy*100:.3f}%")

        # Get the accuracy of the ensemble using a simple majority vote
        ensemble_votes = task0_classifier_ensemble.ensemble_majority_vote(ensemble_output, y[test_index])
        maj_accuracy = task0_classifier_ensemble.accuracy_score(y[test_index], ensemble_votes)
        print(f"Majority vote accuracy= {maj_accuracy * 100:.3f}%")

        fold_number += 1

    # Print the average ensemble accuracy
    # print(f"AVERAGE MAJORITY VOTE ACCURACY= {np.mean(ensemble_accuracies) * 100:.3f}% ± {np.std(ensemble_accuracies):.3f}")