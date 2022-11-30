import numpy as np
import pandas as pd


# Pre-process the data file
def bupa_liver_disorders(csv_file, display_mode=False):

    # Declare the column names
    col_names = ["mcv",
                 "alkphos",
                 "sgpte",
                 "sgot",
                 "gammagt",
                 "drinks",
                 "selector"]

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file, names=col_names)

    # Replace the target variable with 2->1, 1->0
    df['class'] = df['selector'].map({2: 1, 1: 0})
    df.drop(columns=['selector'], inplace=True)
    print(df['class'].value_counts())

    # Return the pandas dataframe
    return df


# Pre-process the data file
def chess(csv_file, display_mode=False):

    # Declare the column names
    col_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
                 "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
                 "36", "winnable"]

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file, names=col_names)

    # Replace the target variable with 2->1, 1->0
    df['class'] = df['winnable'].map({"won": 1, "nowin": 0})
    df.drop(columns=['winnable'], inplace=True)
    print(df['class'].value_counts())

    # Turn all data columns into categoricals
    df = pd.get_dummies(data=df, columns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
                 "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
                 "36"])
    # for col in df.columns:
    #     if col != 'class':
    #         df[col] = df[col].astype('category')
    # print(df.dtypes)

    # Return the pandas dataframe
    return df


# Pre-process the data file
def breast_cancer_wisconsin(csv_file, display_mode=False):

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
    df['Malignant'] = df['Class'].map({4: 1, 2: 0})
    df.drop(columns=['Class'], inplace=True)
    print(df['Malignant'].value_counts())

    # Return the pandas dataframe
    return df
