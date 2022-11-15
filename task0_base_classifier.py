import numpy as np
from utilities import accuracy, data_prep
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# Performs a 5-Fold CV of the classifier
def calculate_clf_accuracy(clf, X, y):
    to_return = []

    # Begin 5-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X, y):
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict(X[test_index])
        to_return += [accuracy.accuracy_score(y[test_index], y_pred)]

    return to_return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # pre-process the data and prepare np arrays...
    # ... for Breast Cancer Wisconsin
    df = data_prep.breast_cancer_wisconsin("./datasets/breast-cancer-wisconsin.data")
    X = df.loc[:, df.columns != 'Malignant'].values
    y = df.loc[:, df.columns == 'Malignant'].values.ravel()

    # ... for Liver
    # df = data_prep.bupa_liver_disorders("./datasets/bupa.data")
    # X = df.loc[:, df.columns != 'class'].values
    # y = df.loc[:, df.columns == 'class'].values.ravel()

    # create our classifier
    lr = LogisticRegression()
    accuracies = calculate_clf_accuracy(lr, X, y)

    print(f"CLASSIFIER ACCURACY= {np.mean(accuracies)*100:.3f}% ± {np.std(accuracies):.3f}")


