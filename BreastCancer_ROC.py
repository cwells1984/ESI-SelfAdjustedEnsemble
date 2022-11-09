from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preprocess


# Display OP
def annot(opi, x, y):
    plt.annotate(f"OP{opi}", xy=(x, y), xytext=(.90*x+.1, .80*y), arrowprops=dict(facecolor='lightgray', shrink=1))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # pre-process the data and prepare np arrays
    df_breast = preprocess.breast_cancer_wisconsin("./datasets/breast-cancer-wisconsin.data")
    df_breast = df_breast[:10]
    X = df_breast.loc[:, df_breast.columns != 'Malignant'].values
    y = df_breast.loc[:, df_breast.columns == 'Malignant'].values.ravel()

    # Training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)


    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)
    y_pred_proba = y_pred_proba[:, 1]
    lr_auc = roc_auc_score(y_test, y_pred_proba)
    print(lr_auc)
    print(y_pred_proba)
    acc = 0
    for i in range(len(y_pred)):
        if y_test[i] == y_pred[i]:
            acc += 1
    acc = acc / len(y_pred)
    print(f"Accuracy {acc}")