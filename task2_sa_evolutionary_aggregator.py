from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from utilities import data_prep, lr_ensemble, evolutionary, preprocess
import numpy as np

# GLOBAL SETTINGS HERE
NUM_CLASSIFIERS = 5
FRACTION=0.2
BASE_CLASSIFIER = DecisionTreeClassifier()
N_POP = 100
N_GEN = 50
PROB_CX = 0.0
PROB_MUT = 0.2

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

        # Now evolve an aggregator
        y_expected = preprocess.onehot_encode(y[train_index])
        evolutionary.run(num_classifiers=NUM_CLASSIFIERS, pop_size=N_POP, n_gen=N_GEN, ensemble_output=ensemble_output, y_expected=y_expected)

        fold_number += 1