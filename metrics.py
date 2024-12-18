from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size and y.size > 0
    return np.sum(y_hat == y) / y.size


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size and cls != None
    denom = np.sum(y_hat == cls)
    assert denom > 0
    return np.sum((y_hat == cls) & (y == cls)) / denom


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size and cls != None
    denom = np.sum(y == cls)
    assert denom > 0
    return np.sum((y_hat == cls) & (y == cls)) / denom


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size and y.size > 0
    return np.sqrt(np.sum((y - y_hat) ** 2) / y.size)


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size and y.size > 0
    return np.sum(np.abs(y - y_hat)) / y.size
