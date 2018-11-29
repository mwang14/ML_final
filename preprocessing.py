import pandas as pd
import numpy as np


# def categories_to_numbers(values):
#     replacements = {}
#     for i in range(len(values)):
#         replacements[values[i]] = i
#         print(replacements)
#         return replacements


def map_categorical_to_numerical(data_matrix, column):
    unique = data_matrix[column].unique()
    d = dict([y, x+1] for x, y in enumerate(sorted(set(unique))))
    new_col = []
    for val in data_matrix[column]:
        new_col.append(d[val])
    data_matrix[column] = new_col
    return data_matrix


def gen_X_and_y(data_set):
    # categorical data in training set
    columns = ['proto', 'service', 'state', 'attack_cat']

    for col in columns:
        cleaned_train_data = map_categorical_to_numerical(data_set, col)

    X_train = cleaned_train_data.drop('attack_cat', axis=1)
    y_train = cleaned_train_data['attack_cat']
    return X_train, y_train
    # print(cleaned_train_data.head())
