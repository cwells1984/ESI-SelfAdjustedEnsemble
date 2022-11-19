import numpy as np
from utilities import accuracy, data_prep
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold


# Performs a 5-Fold CV of the classifier
def calculate_clf_accuracy(clf, X, y):
    train_to_return = []
    test_to_return = []

    # Begin 5-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X, y):
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict(X[train_index])
        train_to_return += [accuracy.accuracy_score(y[train_index], y_pred)]
        y_pred = clf.predict(X[test_index])
        test_to_return += [accuracy.accuracy_score(y[test_index], y_pred)]

    return train_to_return, test_to_return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # pre-process the data and prepare np arrays...
    # ... for Breast Cancer Wisconsin
    # df = data_prep.breast_cancer_wisconsin("./datasets/breast-cancer-wisconsin.data")
    # X = df.loc[:, df.columns != 'Malignant'].values
    # y = df.loc[:, df.columns == 'Malignant'].values.ravel()

    # ... for Liver
    # df = data_prep.bupa_liver_disorders("./datasets/bupa.data")
    # X = df.loc[:, df.columns != 'class'].values
    # y = df.loc[:, df.columns == 'class'].values.ravel()

    # ... for Chess
    df = data_prep.chess("./datasets/chess.data")
    X = df.loc[:, df.columns != 'class'].values
    y = df.loc[:, df.columns == 'class'].values.ravel()
    print(X)

    # create our classifier
    dt = DecisionTreeClassifier()
    train_accuracies, test_accuracies = calculate_clf_accuracy(dt, X, y)

    print(f"TRAINING CLASSIFIER ACCURACY= {np.mean(train_accuracies)*100:.3f}% ± {np.std(train_accuracies):.3f}")
    print(f"TESTING CLASSIFIER ACCURACY= {np.mean(test_accuracies)*100:.3f}% ± {np.std(test_accuracies):.3f}")


