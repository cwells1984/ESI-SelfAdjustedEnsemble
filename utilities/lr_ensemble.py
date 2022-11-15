from utilities import accuracy, preprocess
import copy
import numpy as np
import random


# Return the best accuracy score of the classifiers in the ensemble
# ensemble_output[base classifier #][predicted y index]
def calc_best_accuracy(ensemble_output, y_test):
    best_accuracy = -1.0
    for i in range(len(ensemble_output)):
        y_pred = ensemble_output[i]
        current_accuracy = accuracy.accuracy_score(y_test, y_pred)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
    return best_accuracy


# Creates a list of a given number of classifiers
def create_ensemble(num_classifiers, clf_to_clone):
    ensemble = []
    for i in range(num_classifiers):
        ensemble += [copy.deepcopy(clf_to_clone)]
    return ensemble


# Return the ensemble's predictions by taking a majority vote
# Ensemble output is a 3D array:
# ensemble_output[base classifier #][one-hot encoded y index]
# Returns a list of predictions
def ensemble_majority_vote(ensemble_output):
    ensemble_votes = np.zeros((len(ensemble_output[0]), 2))
    for i in range(len(ensemble_output)):
        for j in range(len(ensemble_output[i])):
            max_arg = np.argmax(ensemble_output[i][j])
            ensemble_votes[j][max_arg] += 1
    return np.argmax(ensemble_votes, axis=1)


# Each classifier in the given ensemble makes predictions on the test data X
# returns ensemble_output, a 2D array:
# ensemble_output[base classifier #][predicted y index]
def ensemble_predict(ensemble, X):
    ensemble_output = []
    for clf in ensemble:
        ensemble_output += [clf.predict(X)]
    return ensemble_output


# Get a sample of the training data, with replacement
def get_subsample(fraction, combined_training_data):

    # use random.choices() for sampling
    k = int(fraction * len(combined_training_data))
    subsample = np.array(random.choices(combined_training_data, k=k))

    # keep taking subsamples until we have more than one unique target value
    while len(np.unique(subsample[:, 0])) <= 1:
        subsample = np.array(random.choices(combined_training_data, k=k))
    return subsample


# Takes the 2D ensemble output at transforms it into a 3D one-hot encoded array
# Returns ensemble output, a 3D array:
# ensemble_output_onehot[base classifier #][one-hot encoded y index]
def onehot_ensemble_output(ensemble_output):
    ensemble_output_onehot = []
    for i in range(len(ensemble_output)):
        ensemble_output_onehot += [preprocess.onehot_encode(ensemble_output[i])]
    return ensemble_output_onehot


# Train the input classifier on a subset of training data
def train_classifier_on_subset(fraction, clf, X_train, y_train):
    train_subset = get_subsample(fraction, np.column_stack((y_train, X_train)).tolist())
    clf.fit(train_subset[:, 1:], train_subset[:, 0])