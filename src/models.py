import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import random
from sklearn.preprocessing import LabelEncoder


def alphabetize_ordinals(df, list_ordinals):
    """
    Rename ordinal features of a dataframe based on their order in list_ordinals
    Example: dict_education = {"grad":"A_grad","undergrad":"B_undergrad","high-school":"C_high-school","others":"D_others"}

    list_ordinals: list of ordinal values that require sorting
    :param df: single column dataframe with the ordinal feature selected
    :return: a column dataframe with the feature values encoded
    """
    df = df.to_frame()
    if len(df.columns) != 1:
        raise ValueError("dataframe needs to have a single column")

    dict_rename = dict()
    for i in range(len(list_ordinals)):
        dict_rename[list_ordinals[i]] = str(i) + '__' + list_ordinals[i]

    df = df.replace({df.columns[0]: dict_rename})
    return df


def transform_features(df, dict_features):
    """
    Summary: take a dataframe and dummify its categorical features ('c) and encode its ordinal features ('o')

    :param df: dataframe to be transformed
    :param dict_features: dictionary indicating which features are categorical or ordinal
    :return: transformed df
    """

    labelencoder = LabelEncoder()
    cat_features = list()
    for key, val in dict_features.items():
        if val == 'o':
            df[key] = labelencoder.fit_transform(df[key])
        elif val == 'c':
            cat_features.append(key)
        else:
            raise ValueError('Unidentified feature type. Please either enter "o" or "c"')
    df = pd.get_dummies(df, columns=cat_features, drop_first=True)
    return df

def train_model(df,model_info):
    """
    Summary: train a model and output trained model in a pickle format along with relevant artifacts
    :param df: dataframe following processing
    :param model_info: dictionary containing model information
    :return:
    """
