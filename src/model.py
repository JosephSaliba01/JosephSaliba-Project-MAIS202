# import statmeents
import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

# read .csv file
df = pd.read_csv('star_classification.csv')

# dataframes per class
df_of_class = {
    'GALAXY': df[df['class'] == 'GALAXY'],
    'QSO': df[df['class'] == 'QSO'],
    'STAR': df[df['class'] == 'STAR']
}

# values of filters per class:
filter_values, filter_values_info = {}, {}

for value in {'u', 'g', 'r', 'i', 'z'}:
    filter_values[value], filter_values_info[value] = dict(), dict()

    for stellar_class in {'GALAXY', 'QSO', 'STAR'}:
        filter_values[value][stellar_class] = list(
            df_of_class[stellar_class][value]
        )

        filter_values_info[value][stellar_class] = dict()

        filter_values_info[value][stellar_class]['mean'] = np.mean(
            filter_values[value][stellar_class]
        )

        filter_values_info[value][stellar_class]['std'] = np.std(
            filter_values[value][stellar_class]
        )

# convert df[['u', 'g', 'r', 'i', 'z']] to a list of values
X = df[['u', 'g', 'r', 'i', 'z']].values.tolist()

# convert df['class'] to a list of values
y = df['class'].values.tolist()

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

def generate_model():
    # use random forest to train model
    rf = RandomForestClassifier(n_estimators=100)

    # train model
    rf.fit(X_train, y_train)

    # save model
    dump(rf, 'model.joblib')
    print("Training Complete.")
    return rf


if __name__ == '__main__':
    # print mean and standard deviation of each filter value
    print(json.dumps(filter_values_info, sort_keys=True, indent=4))

    rf = generate_model()

    # print model accuracy
    print('Accuracy:', rf.score(X_test, y_test))

