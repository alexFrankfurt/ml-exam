import numpy as np
import pandas as pd
import tree

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

     def __init__(self, x, y, n_trees, n_features, sample_sz, depth=10, min_leaf=5):
        np.random.seed(12)
        if n_features == 'sqrt':
            self.n_features = int(np.sqrt(x.shape[1]))
        elif n_features == 'log2':
            self.n_features = int(np.log2(x.shape[1]))
        else:
            self.n_features = n_features
        print(self.n_features, "sha: ",x.shape[1])
        self.x, self.y, self.sample_sz, self.depth, self.min_leaf  = x, y, sample_sz, depth, min_leaf
        self.trees = [DecisionTreeClassifier() for i in range(n_trees)]
