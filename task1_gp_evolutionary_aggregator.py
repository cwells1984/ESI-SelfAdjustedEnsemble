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
PROB_CX = 0.0
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


# The fitness function to be maximized
# Creates an output by evaluating the expression tree's inputs and comparing against the list y_expected
# Returns a 1-element tuple of the fitness, here the accuracy compared to y_expected
def evaluate(individual):

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    # Create list tree_output = the predictions of the members of the ensemble
    tree_output = []
    for i in range(len(ensemble_output[0])):
        res = func(ensemble_output[0][i], ensemble_output[1][i], ensemble_output[2][i], ensemble_output[3][i], ensemble_output[4][i])
        #res = func(ensemble_output[0][i], ensemble_output[1][i], ensemble_output[2][i], ensemble_output[3][i], ensemble_output[4][i], ensemble_output[5][i], ensemble_output[6][i], ensemble_output[7][i], ensemble_output[8][i], ensemble_output[9][i])
        if res is None or type(res) is float:
            return 0.0,
        tree_output.append(res)

    # Compute tree's accuracy by comparing tree_output and y_expected
    acc = 0.0
    for i in range(len(tree_output)):
        c1 = np.argmax(tree_output[i])
        c2 = np.argmax(y_expected[i])
        if c1 == c2:
            acc += 1.0
    return acc / len(tree_output),


# Register evolutionary operators
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
toolbox.register("evaluate", evaluate)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # pre-process the data and prepare np arrays...
    # ... for Breast Cancer Wisconsin
    # df = data_prep.breast_cancer_wisconsin("./datasets/breast-cancer-wisconsin.data")
    # X = df.loc[:, df.columns != 'Malignant'].values
    # y = df.loc[:, df.columns == 'Malignant'].values.ravel()

    # ... for Liver
    df = data_prep.bupa_liver_disorders("./datasets/bupa.data")
    X = df.loc[:, df.columns != 'class'].values
    y = df.loc[:, df.columns == 'class'].values.ravel()

    # create the ensemble
    ensemble = lr_ensemble.create_ensemble(num_classifiers=NUM_CLASSIFIERS, clf_to_clone=BASE_CLASSIFIER)

    # accuracies to average across folds
    avg_best_base = []
    agg_accuracies = []
    maj_accuracies = []

    # Begin 5-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    fold_number = 1
    for train_index, test_index in kf.split(X, y):
        print(f"FOLD NUMBER {fold_number}")

        # train each item of the ensemble on a subset of the training data
        for clf in ensemble:
            lr_ensemble.train_classifier_on_subset(FRACTION, clf, X[train_index], y[train_index])

        # take the output of the ensembles' predictions on the training data
        # this is how we will train the aggregator, the test data will be untouched
        ensemble_output = lr_ensemble.ensemble_predict(ensemble, X[train_index])
        ensemble_output = lr_ensemble.onehot_ensemble_output(ensemble_output)

        # Initialize the population and stats we will track
        pop = toolbox.population(n=N_POP)
        hof = tools.HallOfFame(20)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Using the predictions made on the training data, evolve the aggregator
        y_expected = preprocess.onehot_encode(y[train_index])
        algorithms.eaSimple(pop, toolbox, cxpb=PROB_CX, mutpb=PROB_MUT, ngen=N_GEN, stats=stats, halloffame=hof, verbose=True)
        print(hof[0])

        # Now that the aggregators have evolved, we will have the base classifiers make predictions on the test data
        # These predictions will then be evaluated on the top-ranked aggregator
        ensemble_output = lr_ensemble.ensemble_predict(ensemble, X[test_index])
        best_base = lr_ensemble.calc_best_accuracy(ensemble_output, y[test_index])
        avg_best_base += [best_base]
        print(f"Best base classifier accuracy= {best_base*100:.3f}%")

        # One-hot encode the ensemble output that we will be inputting to the aggregator
        ensemble_output = lr_ensemble.onehot_ensemble_output(ensemble_output)

        # Get the accuracy of the ensemble using a simple majority vote
        ensemble_votes = lr_ensemble.ensemble_majority_vote(ensemble_output)
        maj_accuracy = accuracy.accuracy_score(y[test_index], ensemble_votes)
        maj_accuracies += [maj_accuracy]
        print(f"Majority vote accuracy= {maj_accuracy * 100:.3f}%")

        # feed these estimates into our best aggregator and get the aggregator's accuracy
        y_expected = preprocess.onehot_encode(y[test_index])
        agg_accuracy = evaluate(hof[0])[0]
        agg_accuracies += [agg_accuracy]
        print(f"Evolved aggregator accuracy= {agg_accuracy*100:.3f}%")

        fold_number += 1

    # Print the average accuracy of the majority vote and evolved aggregator across folds
    print("==============================")
    print(f"AVERAGE BEST BASE ACCURACY= {np.mean(avg_best_base) * 100:.3f}% ± {np.std(avg_best_base):.3f}")
    print(f"AVERAGE MAJORITY VOTE ACCURACY= {np.mean(maj_accuracies) * 100:.3f}% ± {np.std(maj_accuracies):.3f}")
    print(f"AVERAGE AGGREGATOR VOTE ACCURACY= {np.mean(agg_accuracies) * 100:.3f}% ± {np.std(agg_accuracies):.3f}")