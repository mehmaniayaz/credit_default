import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import random


def generate_cat_num_dataframe(n_samples=100, n_features=10, n_cats=2):
    '''
    Summary: generate a random dataframe with a mix of categorical and numerical variables and a single
    binary target variable.
    :param n_samples: number of samples
    :param n_features: number of features
    :param n_cats: number of categorical features
    :return: dataframe
    '''
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=2)
    df = pd.DataFrame(X, columns=['feature_' + str(x) for x in range(np.shape(X)[1])])
    cat_feature_list = random.sample(list(df.columns), n_cats)
    df['target'] = y

    bins_list = [3, 4, 5]
    dict_col = {}
    for feature in cat_feature_list:
        dict_col[feature] = 'cat_' + feature
        bin_number = random.sample(bins_list, 1)[0]
        df[feature] = pd.cut(df[feature], bins=bin_number, labels=[str(x) for x in range(bin_number)])
    df.rename(columns=dict_col, inplace=True)
    return df
