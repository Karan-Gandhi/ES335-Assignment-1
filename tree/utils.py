"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    Will return True if the given series is real
    """
    return np.sum(i in str(y.size) for i in ['int']) > 0


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    p = Y.value_counts(normalize=True)
    S = -np.sum(p * np.log2(p))
    return S


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    p = Y.value_counts(normalize=True)
    return 1 - np.sum(p * p)

def mse(Y: pd.Series) -> float:
    """
    Function to calculate mse of data
    """
    return np.mean((Y - np.mean(Y)) ** 2)

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if criterion.lower() not in ['entropy', 'gini', 'mse']:
        raise ValueError('Choose a valid criterion: entropy, gini, or mse')
    
    match criterion.lower():
        case 'entropy':
            parent_impurity = entropy(Y)
            if check_ifreal(attr):
                # continue from here
                pass
    return

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    best_information_gain = -np.inf
    best_attr = None

    for feature in features:
        current_information_gain = information_gain(y, X[feature], criterion)
        if current_information_gain > best_information_gain:
            best_information_gain = current_information_gain
            best_attr = feature

    return best_attr


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """
    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    assert y.size == X.shape[0]
    X_left, y_left, X_right, y_right = None, None, None, None

    X_left, y_left = X[X[attribute] <= value], y[X[attribute] <= value]
    X_right, y_right = X[X[attribute] > value], y[X[attribute] > value]

    return X_left, y_left, X_right, y_right