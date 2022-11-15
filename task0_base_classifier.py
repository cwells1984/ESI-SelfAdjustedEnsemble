import numpy as np
from utilities import accuracy, data_prep
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # pre-process the data and prepare np arrays
    df_liver = data_prep.bupa_liver_disorders("./datasets/bupa.data")
    X = df_liver.loc[:, df_liver.columns != 'class'].values
    y = df_liver.loc[:, df_liver.columns == 'class'].values.ravel()

    # create our classifier
    lr = LogisticRegression()
    accuracies = []

    # Begin 5-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    fold_number = 1
    for train_index, test_index in kf.split(X, y):
        lr.fit(X[train_index], y[train_index])
        y_pred = lr.predict(X[test_index])
        accuracies += [accuracy.accuracy_score(y[test_index], y_pred)]
        fold_number += 1

    print(f"CLASSIFIER ACCURACY= {np.mean(accuracies)*100:.3f}% ± {np.std(accuracies):.3f}")


