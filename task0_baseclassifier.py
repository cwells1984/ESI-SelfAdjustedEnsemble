import numpy as np
import preprocess
import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def get_subsample(fraction, combined_training_data):
    k = int(fraction * len(combined_training_data))
    #print(f'# to train= {len(combined_training_data)}')
    #print(f'k= {k}')
    subsample = np.array(random.sample(combined_training_data, k=k))
    while len(np.unique(subsample[:, 0])) <= 1:
        subsample = np.array(random.sample(combined_training_data, k=k))
    #print(f'# in subsample= {len(subsample)}')
    return subsample


def accuracy_score(expected, predicted):
    accuracy = 0
    for i in range(len(expected)):
        if predicted[i] == expected[i]:
            accuracy += 1
    accuracy = accuracy / len(expected)
    return accuracy


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # pre-process the data and prepare np arrays
    df_liver = preprocess.bupa_liver_disorders("./datasets/bupa.data")
    X = df_liver.loc[:, df_liver.columns != 'class'].values
    y = df_liver.loc[:, df_liver.columns == 'class'].values.ravel()

    # Create our base classifier
    clf = LogisticRegression()

    # Using 5-fold cross validation calculate the accuracy of the classifier
    accuracies = []
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    fold_number = 1
    for train_index, test_index in kf.split(X, y):
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict(X[test_index])
        accuracies += [accuracy_score(y[test_index], y_pred)]
        print(f"Fold #{fold_number}, Accuracy= {accuracies[-1]*100:.3f}%")
    print(f"5-Fold CV Accuracy= {np.mean(accuracies)*100:.3f}% ± {np.std(accuracies):.3f}")
