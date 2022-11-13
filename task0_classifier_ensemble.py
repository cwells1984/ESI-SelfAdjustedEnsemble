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
    #subsample = np.array(random.sample(combined_training_data, k=k))
    subsample = np.array(random.choices(combined_training_data, k=k))
    while len(np.unique(subsample[:, 0])) <= 1:
        #subsample = np.array(random.sample(combined_training_data, k=k))
        subsample = np.array(random.choices(combined_training_data, k=k))
    #print(f'# in subsample= {len(subsample)}')
    return subsample


def train_classifier_on_subset(clf, fraction, X_train, y_train):
    train_subset = get_subsample(fraction, np.column_stack((y_train, X_train)).tolist())
    #print(f"Training subset length= {len(train_subset)}")
    clf.fit(train_subset[:, 1:], train_subset[:, 0])


def accuracy_score(expected, predicted):
    accuracy = 0
    for i in range(len(expected)):
        if predicted[i] == expected[i]:
            accuracy += 1
    accuracy = accuracy / len(expected)
    return accuracy


def execute_and_return_ensemble_output():
    # pre-process the data and prepare np arrays
    df_liver = preprocess.bupa_liver_disorders("./datasets/bupa.data")
    X = df_liver.loc[:, df_liver.columns != 'class'].values
    y = df_liver.loc[:, df_liver.columns == 'class'].values.ravel()

    # create the ensemble
    num_classifiers = 20
    fraction = 0.1
    ensemble = []
    for i in range(num_classifiers):
        ensemble += [LogisticRegression()]

    # Using 5-fold cross validation calculate the accuracy of the ensemble
    accuracies = []
    ensemble_accuracies = []
    best_base_accuracies = []
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    fold_number = 1
    for train_index, test_index in kf.split(X, y):
        print(f"Fold Number {fold_number}")
        # print(f"Master training set length= {len(train_index)}")

        # train each item of the ensemble on a subset of the training data
        for clf in ensemble:
            train_classifier_on_subset(clf, fraction, X[train_index], y[train_index])

        # make predictions for each base classifier
        accuracies = []
        ensemble_output = []
        for clf in ensemble:
            y_pred = clf.predict(X[test_index])
            ensemble_output += [y_pred]
            accuracy = accuracy_score(y[test_index], y_pred)
            accuracies += [accuracy]
            # print(f"Base classifier accuracy= {accuracy*100:.3f}%")

        print(f"Best base classifier accuracy= {np.max(accuracies) * 100:.3f}%")
        best_base_accuracies += [np.max(accuracies)]
        print(f"Average base classifier accuracy= {np.mean(accuracies) * 100:.3f}% ± {np.std(accuracies):.3f}")

        # Conduct a simple majority vote
        ensemble_votes = np.zeros((len(y[test_index]), 2))
        for i in range(len(ensemble_output)):
            for j in range(len(ensemble_output[i])):
                if ensemble_output[i][j] == 0:
                    ensemble_votes[j][0] += 1
                else:
                    ensemble_votes[j][1] += 1
        ensemble_votes = np.argmax(ensemble_votes, axis=1)
        ensemble_accuracies += [accuracy_score(y[test_index], ensemble_votes)]
        print(f"Ensemble accuracy= {ensemble_accuracies[-1] * 100:.3f}%")

        fold_number += 1

    # Print the average ensemble accuracy
    print(f"AVERAGE ENSEMBLE ACCURACY= {np.mean(ensemble_accuracies) * 100:.3f}% ± {np.std(ensemble_accuracies):.3f}")
    print(
        f"AVERAGE BEST BASE ACCURACY= {np.mean(best_base_accuracies) * 100:.3f}% ± {np.std(best_base_accuracies):.3f}")
    return ensemble_output


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    execute_and_return_ensemble_output()