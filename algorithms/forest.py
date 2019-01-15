import numpy as np
import pandas as pd
from tree import DecisionTreeClassifier
from random import sample

class RandomForestRegressor():
    # independent variables of training set
    #x = 1

    # dependent variables
    #y =
    # n_trees - number of uncorrelated trees we ensemble to create the random forest
    # n_features - the number of features to sample and pass onto each tree
    # sample_size - the number of rows randomly selected and passed onto each tree
    # depth - depth of each decision tree
    # min_leaf - minimum number of rows required in a node to cause further split

     def __init__(self, x, y, n_trees, n_features, sample_sz, depth=10, min_leaf=2):
         tree_list = []
         for i in range(0, n_trees):
            random_features = []
            random_rows     = []
            random_y        = []

            rand_feature_idxs = sample(range(0, len(x[0])), n_features)
            for features in x:
                random_features.append([features[i] for i in rand_feature_idxs])

            rand_row_idxs = sample(range(0, len(x)), sample_sz)
            for idx, features in enumerate(x):
                if idx in rand_row_idxs:
                    random_rows.append(features)
                    random_y.append(y[idx])


            tree = DecisionTreeClassifier(depth, min_leaf)
            tree.fit(random_rows, random_y)
            tree_list.append(tree)
