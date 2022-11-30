from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utilities import accuracy, data_prep, evolutionary_sa, lr_ensemble, preprocess
import numpy as np

# GLOBAL SETTINGS HERE
NUM_CLASSIFIERS = 100
FRACTION=0.35
BASE_CLASSIFIER = DecisionTreeClassifier()
N_POP = 100
N_GEN = 25

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # pre-process the data and prepare np arrays...
    # ... for Breast Cancer Wisconsin
    # df = data_prep.breast_cancer_wisconsin("./datasets/breast-cancer-wisconsin.data")
    # X = df.loc[:, df.columns != 'Malignant'].values
    # y = df.loc[:, df.columns == 'Malignant'].values.ravel()

    # ... for Chess
    # df = data_prep.chess("./datasets/chess.data")
    # X = df.loc[:, df.columns != 'class'].values
    # y = df.loc[:, df.columns == 'class'].values.ravel()

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
    avg_weights = []
    std_weights = []

    # create the evolutionary class to run
    e = evolutionary_sa.Evolutionary(num_classifiers=NUM_CLASSIFIERS)

    # Begin 5-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    fold_number = 1
    for train_index, test_index in kf.split(X, y):
        print(f"FOLD NUMBER {fold_number}")
        X_base, X_agg, y_base, y_agg = train_test_split(X[train_index], y[train_index], test_size=0.2)

        # train each item of the ensemble on a subset of the training data
        for clf in ensemble:
            lr_ensemble.train_classifier_on_subset(FRACTION, clf, X_base, y_base)

        # take the output of the ensembles' predictions on the training data
        # this is how we will train the aggregator, the test data will be untouched
        ensemble_output = lr_ensemble.ensemble_predict(ensemble, X_agg)
        ensemble_output = lr_ensemble.onehot_ensemble_output(ensemble_output)

        # Now evolve a aggregator
        y_expected = preprocess.onehot_encode(y_agg)
        hall_of_fame = e.run(pop_size=N_POP, n_gen=N_GEN, ensemble_output=ensemble_output, y_expected=y_expected)
        print(hall_of_fame[0])
        weights = hall_of_fame[0:NUM_CLASSIFIERS]
        avg_weights += [np.mean(weights)]
        std_weights += [np.std(weights)]

        # Now that the aggregators have evolved, we will have the base classifiers make predictions on the test data
        # These predictions will then be evaluated on the top-ranked aggregator
        ensemble_output = lr_ensemble.ensemble_predict(ensemble, X[test_index])
        best_base = lr_ensemble.calc_best_accuracy(ensemble_output, y[test_index])
        avg_best_base += [best_base]
        print(f"Best base classifier accuracy= {best_base * 100:.3f}%")

        # One-hot encode the ensemble output that we will be inputting to the aggregator
        ensemble_output = lr_ensemble.onehot_ensemble_output(ensemble_output)

        # Get the accuracy of the ensemble using a simple majority vote
        ensemble_votes = lr_ensemble.ensemble_majority_vote(ensemble_output)
        maj_accuracy = accuracy.accuracy_score(y[test_index], ensemble_votes)
        maj_accuracies += [maj_accuracy]
        print(f"Majority vote accuracy= {maj_accuracy * 100:.3f}%")

        # feed these estimates into our best aggregator and get the aggregator's accuracy
        e.ensemble_output = ensemble_output
        e.y_expected = preprocess.onehot_encode(y[test_index])
        agg_accuracy = e.evaluate(hall_of_fame[0])[0]
        agg_accuracies += [agg_accuracy]
        print(f"Evolved aggregator accuracy= {agg_accuracy * 100:.3f}%")

        fold_number += 1

    # Print the average accuracy of the majority vote and evolved aggregator across folds
    print("==============================")
    print(f"AVERAGE BEST BASE ACCURACY= {np.mean(avg_best_base) * 100:.3f}% ± {np.std(avg_best_base):.3f}")
    print(f"AVERAGE MAJORITY VOTE ACCURACY= {np.mean(maj_accuracies) * 100:.3f}% ± {np.std(maj_accuracies):.3f}")
    print(f"AVERAGE AGGREGATOR VOTE ACCURACY= {np.mean(agg_accuracies) * 100:.3f}% ± {np.std(agg_accuracies):.3f}")
    print(f"AVERAGE MEAN/STD OF WEIGHTS= {np.mean(avg_weights):.3f}, {np.mean(std_weights):.3f}")
