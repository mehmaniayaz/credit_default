import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


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


def train_model(df, model_info):
    """
    Summary: train a model and output trained model in a pickle format along with relevant artifacts
    :param df: dataframe following processing
    :param model_info: dictionary containing model information
    :return:
    """

    features = model_info['features']
    targets = model_info['targets']
    test_size = model_info['test_size']
    random_state = model_info['random_state']

    X = np.array(df[features])
    y = np.array(df[targets])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    clf_scorer = make_scorer(score_func=balanced_accuracy_score, greater_is_better=True)
    parameters = {'n_estimators': [100, 200, 600],
                  'max_depth': [2, 5, 10, 100],
                  'min_samples_split': [2, 4, 6]}
    clf = RandomizedSearchCV(RandomForestClassifier(), parameters, scoring=clf_scorer)

    clf.fit(X_train, y_train)

    if model_info['plot_confusion_matrix']:
        # annotate overfitting score in the confusion matrix plot
        over_fitting_score = clf.predict(X_test) / clf.predict(X_train)
        plt.figure(figsize=(10, 10))
        plot_confusion_matrix(clf, X_test, y_test, display_labels=['No', 'Yes'], normalize='true')
        plt.title('Test set. Overfitting score is: {}'.format(over_fitting_score))
        plt.savefig('../results/confusion-matrix-test.png')
        plt.close()

        plt.figure(figsize=(10, 10))
        plot_confusion_matrix(clf, X_train, y_train, display_labels=['No', 'Yes'], normalize='true')
        plt.title('Training set. Overfitting score is: {}'.format(over_fitting_score))
        plt.savefig('../results/confusion-matrix-train.png')
        plt.close()
