import numpy as np
from utilities import accuracy, data_prep
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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
    # df = data_prep.breast_cancer_wisconsin("./datasets/breast-cancer-wisconsin.data")
    # X = df.loc[:, df.columns != 'Malignant'].values
    # y = df.loc[:, df.columns == 'Malignant'].values.ravel()

    # ... for Liver
    df = data_prep.bupa_liver_disorders("./datasets/bupa.data")
    X = df.loc[:, df.columns != 'class'].values
    y = df.loc[:, df.columns == 'class'].values.ravel()

    # Create a grid of tuneable parameters to test
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Create the classifier
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
    rf_random.fit(X, y)
    print(rf_random.best_params_)

    # Run the default classifier
    # rf = RandomForestClassifier()
    # accuracies = calculate_clf_accuracy(rf, X, y)
    # print(f'DEFAULT CLASSIFIER ACCURACY={np.mean(accuracies):.3f} {chr(177)}{np.std(accuracies):.3f}')
    #
    # # create our classifier
    # rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, min_samples_leaf=2, max_depth=30, bootstrap=True)
    # accuracies = calculate_clf_accuracy(rf, X, y)
    # print(f"TUNED CLASSIFIER ACCURACY= {np.mean(accuracies)*100:.3f}% ± {np.std(accuracies):.3f}")


