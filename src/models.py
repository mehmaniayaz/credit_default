import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import random

def test_alphabetize_ordinals(df,list_ordinals):
    """
    Rename ordinal features of a dataframe based on their order in list_ordinals
    Example: dict_education = {"grad":"A_grad","undergrad":"B_undergrad","high-school":"C_high-school","others":"D_others"}


    :param df: single column dataframe with the ordinal feature selected
    :param list_ordinals: values of the ordinal feature
    :return: a column dataframe with the feature values encoded
    """
