# import statmeents
import numpy as np
import pandas as pd
import matplotlib as plt
import json

# read .csv file
df = pd.read_csv('star_classification.csv')

# dataframes per class

dataframes_per_class = {
    'GALAXY': df[df['class'] == 'GALAXY'],
    'QSO': df[df['class'] == 'QSO'],
    'STAR': df[df['class'] == 'STAR']
}

# values of filters per class:
filter_values, filter_values_info = {}, {}

for value in {'u', 'g', 'r', 'i', 'z'}:
    filter_values[value] = {}
    filter_values_info[value] = {}

    for stellar_class in {'GALAXY', 'QSO', 'STAR'}:
        filter_values[value][stellar_class] = list(
            dataframes_per_class[stellar_class][value])

        filter_values_info[value][stellar_class] = {}

        filter_values_info[value][stellar_class]['mean'] = np.mean(
            filter_values[value][stellar_class])

        filter_values_info[value][stellar_class]['std'] = np.std(
            filter_values[value][stellar_class])

print(json.dumps(filter_values_info, sort_keys=True, indent=4))

# train using potential models


# train


# test
