import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import random


def alphabetize_ordinals(df,list_ordinals):
    """
    Rename ordinal features of a dataframe based on their order in list_ordinals
    Example: dict_education = {"grad":"A_grad","undergrad":"B_undergrad","high-school":"C_high-school","others":"D_others"}

    list_ordinals: list of ordinal values that require sorting
    :param df: single column dataframe with the ordinal feature selected
    :return: a column dataframe with the feature values encoded
    """

    if len(df.columns) != 1:
        raise ValueError("dataframe needs to have a single column")

    dict_rename = dict()
    for i in range(len(list_ordinals)):
        dict_rename[list_ordinals[i]] = str(i) + '__' + list_ordinals[i]

    df = df.replace({df.columns[0]: dict_rename})
    return df
