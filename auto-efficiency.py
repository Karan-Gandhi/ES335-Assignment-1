import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from tree.utils import *
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn
data = data[data['horsepower'] != '?']
data['horsepower'] = data['horsepower'].astype(float)
data['car name'] = pd.Categorical(data['car name']).codes

def prepare_dataset(X, y):
    total_size = len(X)
    train_size = int(total_size * 0.7)

    X_train = pd.DataFrame(X[:train_size])
    y_train = pd.Series(y[:train_size])
    X_test = pd.DataFrame(X[train_size:])
    y_test = pd.Series(y[train_size:])

    return X_train, y_train, X_test, y_test

X = data.drop(columns=['mpg'])
y = data['mpg']

X_train, y_train, X_test, y_test = prepare_dataset(X, y)

def print_report(y, y_pred, model_name):
    print(f"{model_name} RMSE: ", rmse(y_pred, y))
        
def evaluate_tree(X_train, y_train, X_test, y_test):
    tree = DecisionTree(criterion='mse', max_depth=10)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print_report(y_test.to_numpy(), y_pred.to_numpy(), "Decision Tree Scratch")
    
def evaluate_tree_sklearn(X_train, y_train, X_test, y_test):
    tree = DecisionTreeRegressor(max_depth=10)
    tree.fit(one_hot_encoding(X_train).to_numpy(), y_train)
    y_pred = tree.predict(one_hot_encoding(X_test).to_numpy())
    print_report(y_test, y_pred, "Decision Tree Sklearn")
    
evaluate_tree(X_train, y_train, X_test, y_test)
evaluate_tree_sklearn(X_train, y_train, X_test, y_test)