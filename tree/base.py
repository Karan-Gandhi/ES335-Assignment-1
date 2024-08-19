"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

@dataclass
class Node:
    children: list
    criterion: str
    depth: int
    samples_x: pd.DataFrame
    samples_y: pd.Series
    split_feature: str
    output_type: Literal['real', 'discrete']
    
    def __init__(self, criterion, depth, output_type):
        self.children = []
        self.criterion = criterion
        self.depth = depth
        self.samples_x = None
        self.samples_y = None
        self.split_feature = None
        self.split_value = None
        self.output_type = output_type
        
    def split(self, max_depth, eps):
        assert (self.samples_x is not None) and (self.samples_y is not None)

        if self.depth == max_depth:
            return
        
        if get_impurity_function(self.criterion)(self.samples_y) <= eps:
            return
        
        # if we have pure data then stop splitting
        # else split into two children based on the best information gain

        best_gain = 0
        opt_split = None
        opt_feature = None

        for feature in self.samples_x.columns:
            info_gain, current_opt_split = information_gain(self.samples_y, self.samples_x[feature], self.criterion)
            
            if info_gain > best_gain:
                opt_split = current_opt_split
                best_gain = info_gain
                opt_feature = feature
                
        assert (opt_split is not None) and (opt_feature is not None)
        
        # Now based on the given feature and the given split value divide into two children
        self.split_feature = opt_feature
        self.split_value = opt_split
        
        left_child_mask = self.samples_x[self.samples_x[opt_feature] <= opt_split]
        right_child_mask = self.samples_x[self.samples_x[opt_feature] > opt_split]
        
        self.children = [Node(self.criterion, self.depth + 1, self.output_type), Node(self.criterion, self.depth + 1, self.output_type)]
        
        self.children[0].add_samples(self.samples_x[left_child_mask], self.samples_y[left_child_mask])
        self.children[1].add_samples(self.samples_x[right_child_mask], self.samples_y[right_child_mask])
        
        self.children[0].split(max_depth, eps)
        self.children[1].split(max_depth, eps)
    
    def add_samples(self, samples_x, samples_y):
        self.samples_x = samples_x
        self.samples_y = samples_y
    
    def predict(self, X: pd.DataFrame):
        if len(self.children) == 0:
            # check if it is real output or discrete output
            if self.output_type == 'discrete':
                pass
            else:
                pass
            
        

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to
    root: Node
    eps: float

    def __init__(self, criterion, max_depth=5, eps=1e-3):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = Node(criterion, 0)
        self.eps = eps

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 
        self.root.add_samples(X, y)
        self.root.split(self.max_depth, self.eps)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        pass

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass
