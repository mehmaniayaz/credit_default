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
from sklearn.model_selection import learning_curve
import pickle
import os


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
        elif val != 'n':
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
    if os.path.isdir('../results/' + model_info['model_name']):
        raise ValueError('A model with this name already exists!')
    else:
        os.makedirs('../results/' + model_info['model_name'])

    features = model_info['features']
    targets = model_info['targets']
    test_size = model_info['test_size']
    random_state = model_info['random_state']
    parameters = model_info['parameters']

    X = np.array(df[features])
    y = np.array(df[targets])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    clf_scorer = make_scorer(score_func=balanced_accuracy_score, greater_is_better=True)

    clf = RandomizedSearchCV(RandomForestClassifier(), parameters, scoring=clf_scorer)

    clf.fit(X_train, y_train)

    pickle.dump(clf, open('../results/' + model_info['model_name'] + '/trained_model.sav', 'wb'))

    if model_info['plot_confusion_matrix']:
        # annotate overfitting score in the confusion matrix plot
        over_fitting_score = balanced_accuracy_score(y_test, clf.predict(X_test)) / balanced_accuracy_score(y_train,
                                                                                                            clf.predict(
                                                                                                                X_train))
        testing_score = balanced_accuracy_score(y_test, clf.predict(X_test))
        training_score = balanced_accuracy_score(y_train, clf.predict(X_train))
        fig = plt.figure(figsize=(10, 10), facecolor='white')
        plot_confusion_matrix(clf, X_test, y_test, display_labels=['No', 'Yes'], normalize='true')
        plt.title('Test set. Overfitting score is: {}\nTraining score is: {}'.format(round(over_fitting_score, 2),
                                                                                     round(training_score, 2)))
        plt.tight_layout()
        plt.savefig('../results/' + model_info['model_name'] + '/confusion-matrix-test.png',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()

        fig = plt.figure(figsize=(10, 10), facecolor='white')
        plot_confusion_matrix(clf, X_train, y_train, display_labels=['No', 'Yes'], normalize='true')
        plt.title('Training set. Overfitting score is: {}\nTesting score is: {}'.format(round(over_fitting_score, 2),
                                                                                        round(testing_score, 2)))
        plt.tight_layout()
        plt.savefig('../results/' + model_info['model_name'] + '/confusion-matrix-train.png',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()

    if model_info['learning_curve']:
        train_size, train_scores, test_scores, fit_times, score_time = learning_curve(estimator=clf, X=X_train,
                                                                                      y=y_train, scoring=make_scorer(
                balanced_accuracy_score), return_times=True)

        fig = plt.figure(figsize=(10, 10), facecolor='white')
        plt.scatter(train_size / len(y_train), np.mean(train_scores, axis=1), c='b',
                    label='training')
        plt.scatter(train_size / len(y_train), np.mean(test_scores, axis=1), c='r', label='testing')
        plt.xlabel('training size (% of total samples)', fontsize=15)
        plt.ylabel('score', fontsize=15)
        plt.legend(prop={"size":15})
        plt.tight_layout()
        plt.savefig('../results/' + model_info['model_name'] + '/learning_curve.png', facecolor=fig.get_facecolor(),
                    edgecolor='none')
        plt.close()
