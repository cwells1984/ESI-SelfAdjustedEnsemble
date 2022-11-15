# Gets the accuracy score, comparing between a list of expected and predicted values
def accuracy_score(expected, predicted):
    accuracy = 0
    for i in range(len(expected)):
        if predicted[i] == expected[i]:
            accuracy += 1
    accuracy = accuracy / len(expected)
    return accuracy
