# One-hot encodes a given list, assuming the list's values are 0 and 1
def onehot_encode(l):
    l_onehot = []
    for i in range(len(l)):
        if l[i] == 0:
            l_onehot += [[1, 0]]
        else:
            l_onehot += [[0, 1]]
    return l_onehot
