import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


# Pre-process the data file
def preprocess(csv_file, display_mode=False):

    # Declare the column names
    col_names = ["Sample code number",
                 "Clump Thickness",
                 "Uniformity of Cell Size",
                 "Uniformity of Cell Shape",
                 "Marginal Adhesion",
                 "Single Epithelial Cell Size",
                 "Bare Nuclei",
                 "Bland Chromatin",
                 "Normal Nucleoli",
                 "Mitoses",
                 "Class"]

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file, names=col_names)

    # Drop the Sample code number
    df.drop(columns=['Sample code number'], inplace=True)

    # Show missing data if in display mode
    if display_mode:
        print("Data types:")
        print(df.dtypes)
        n_missing = len(df.loc[df['Bare Nuclei'] == '?'])
        n_null = df['Bare Nuclei'].isnull().any()
        print(f'{n_missing} ? values in Bare Nuclei')
        print(f'Are null values in Bare Nuclei? {n_null}\n')

    # Replace the '?' missing values with NaN
    df.replace("?", np.nan, inplace=True)

    # Per the .names file, all the columns will have integer values
    # So, convert all the columns to the numpy Int64 data type to allow for NaN values in the column
    for col in df.columns:
        df[col] = df[col].astype('Int64')

        # Show missing data if in display mode - all columns should be ints and there should be null values
        if display_mode:
            print("Data types:")
            print(df.dtypes)
            n_missing = len(df.loc[df['Bare Nuclei'] == '?'])
            n_null = df['Bare Nuclei'].isnull().any()
            print(f'{n_missing} ? values in Bare Nuclei')
            print(f'Are null values in Bare Nuclei? {n_null}\n')

    # Replace NaNs with column mean
    column_mean = np.int64(df['Bare Nuclei'].mean())
    df['Bare Nuclei'].fillna(column_mean, inplace=True, downcast='infer')

    # Show missing data if in display mode - there should be no more null values
    if display_mode:
        n_null = df['Bare Nuclei'].isnull().any()
        print(f'Are null values in Bare Nuclei? {n_null}\n')

    # Replace the target variable values with 2=>"False" and 4=>"True"
    df['Malignant'] = df['Class'].map({4: True, 2: False})
    df.drop(columns=['Class'], inplace=True)
    print(df['Malignant'].value_counts())

    # Return the pandas dataframe
    return df


# 5-fold CV evaluation of a classifier from module 6 notes
def eval_classifier(clf, X, y):
    acc = []
    kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    for train_index, test_index in kf.split(X, y):
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict(X[test_index])
        fold_acc = accuracy_score(y[test_index], y_pred)
        acc += [fold_acc]
        print(f'Fold accuracy={np.mean(fold_acc):.3f}')
    return np.array(acc)


if __name__ == '__main__':

    # Pre-process the data
    df = preprocess("./datasets/breast-cancer-wisconsin.data")

    # Prepare the input X and y
    X = df.loc[:, df.columns != 'Malignant'].values
    y = df.loc[:, df.columns == 'Malignant'].values.ravel()

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
    #rf = RandomForestClassifier()
    #rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=1000, cv=5, verbose=2, random_state=42, n_jobs=-1)
    #rf_random.fit(X, y)
    #print(rf_random.best_params_)

    # Create a tuned classifier
    # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    # best parameters found above:
    # {'n_estimators': 1600, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': 80, 'bootstrap': False}
    rf_tuned = RandomForestClassifier(n_estimators=400, min_samples_split=10, min_samples_leaf=4, max_depth=100, bootstrap=True)

    # Run the default classifier
    acc = eval_classifier(RandomForestClassifier(), X, y)
    print(f'Default Random Forest accuracy={np.mean(acc):.3f} {chr(177)}{np.std(acc):.3f}')

    # Run the tuned classifier
    acc = eval_classifier(rf_tuned, X, y)
    print(f'Tuned Random Forest accuracy={np.mean(acc):.3f} {chr(177)}{np.std(acc):.3f}')