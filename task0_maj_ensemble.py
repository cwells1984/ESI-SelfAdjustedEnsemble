import numpy as np
from utilities import accuracy, data_prep
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from utilities import accuracy, lr_ensemble, data_prep, preprocess

BASE_CLASSIFIER = DecisionTreeClassifier()


# Performs a 5-Fold CV of the classifier
def calculate_clf_accuracy(ensemble, X, y):
    base_accuracies = []
    accuracies = []

    # Begin 5-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X, y):

        ensemble.fit(X, y)

        # Now find the best base classifier
        best_base_acc = 0
        for clf in ensemble.estimators_:
            y_pred = clf.predict(X[test_index])
            base_acc = accuracy.accuracy_score(y[test_index], y_pred)
            if base_acc > best_base_acc:
                best_base_acc = base_acc
        base_accuracies += [best_base_acc]

        # Now make predictions for the ensemble
        y_pred = ensemble.predict(X[test_index])

        # Now get the accuracy of the majority vote
        maj_accuracy = accuracy.accuracy_score(y[test_index], y_pred)
        accuracies += [maj_accuracy]

    return base_accuracies, accuracies


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

    # create our classifier
    N_values = [100,300,500]
    f_values =[0.2,0.35,0.5]
    for N in N_values:
        for f in f_values:
            ensemble = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=N, max_samples=f)
            base_accuracies, accuracies = calculate_clf_accuracy(ensemble, X, y)
            print(f"N={N}, f={f} & {np.mean(base_accuracies):.3f} ± {np.std(base_accuracies):.3f} & {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f} \\\\")



