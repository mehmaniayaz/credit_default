import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import random


def generate_cat_num_dataframe(n_samples=100, n_features=10, n_cats=2):
    """
    Summary: generate a random dataframe with a mix of categorical and numerical variables and a single
    binary target variable.
    :param n_samples: number of samples
    :param n_features: number of features
    :param n_cats: number of categorical features
    :return: dataframe
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=2)
    df = pd.DataFrame(X, columns=['feature_' + str(x) for x in range(np.shape(X)[1])])
    cat_feature_list = random.sample(list(df.columns), n_cats)
    df['target'] = y

    bins_list = [3, 4, 5]
    dict_col = {}
    for feature in cat_feature_list:
        dict_col[feature] = 'cat_' + feature
        bin_number = random.sample(bins_list, 1)[0]
        df[feature] = pd.cut(df[feature], bins=bin_number, labels=['lbl_' + str(x) for x in range(bin_number)])
    df[cat_feature_list] = df[cat_feature_list].astype('string')
    df.rename(columns=dict_col, inplace=True)
    return df


def encode_string_list(x):
    """
    Summary: take a list of strings x and encode its values to a list of integers.
    :param x: list of strings
    :return y: list of integers that are codes for x. x is sorted before being encoded.

    Example:
    x = ['cat', 'dog', 'book', 'pencil', 'dog', 'book']
    y = [1, 2, 0, 3, 2, 0]
    """
    x_sorted = np.sort(x)
    keys = np.unique(x_sorted)
    values = list(range(len(keys)))
    dict_encode = dict(zip(keys, values))
    y = [dict_encode[x] for x in x]
    return y


def encode_df_cat_columns(df):
    """
    Encode the categorical variables of a dataframe
    :param df: dataframe with numerical and categorical variables
    :return: dataframe with encoded variables
    """
    cat_feature_list = [x for x in df.columns if x not in df._get_numeric_data().columns]
    encode_dict_df = dict()
    for feature in cat_feature_list:
        counter = 0
        val_list = np.unique(df[feature])
        encode_dict_df[feature] = dict()
        for val in val_list:
            encode_dict_df[feature][val] = counter
            counter += 1

    return df.replace(encode_dict_df), encode_dict_df


def reverse_encode_df_cat_columns(df,encode_dict):
    """
    Reverse the encodings of a dataframe back to its categorical variables
    :param df: encoded dataframe
    :param encode_dict: nested dictionary that was used for encoding the original dataframe
    :return: dataframe that is now decoded
    """
    decode_dict={}
    for feature in encode_dict.keys():
        decode_dict[feature] = {v: k for k, v in encode_dict[feature].items()}
    df_decoded = df.replace(decode_dict)
    df_decoded[list(encode_dict.keys())] = df_decoded[encode_dict.keys()].astype('string')
    return df_decoded

    