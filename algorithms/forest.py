import numpy as np
import pandas as pd
import itertools
import operator
from algorithms.tree import DecisionTreeClassifier
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

    def __init__(self, x, y, n_trees, n_features, sample_sz, depth=5, min_leaf=2):
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

        most_frequent = []
        for i in range(0,len(y)):
            most_frequent.append(self.choose_most_frequent(x, tree_list, i))
        self.best_guess = most_frequent



    def choose_most_frequent(self, X, trees, i):
        gueses = []
        for tree in trees:
            gueses.append(tree.predict(X)[i])
        return self.most_common(gueses)


    def most_common(self, L):
        # get an iterable of (item, iterable) pairs
        SL = sorted((x, i) for i, x in enumerate(L))
        # print 'SL:', SL
        groups = itertools.groupby(SL, key=operator.itemgetter(0))
        # auxiliary function to get "quality" for an item
        def _auxfun(g):
            item, iterable = g
            count = 0
            min_index = len(L)
            for _, where in iterable:
                count += 1
                min_index = min(min_index, where)
            # print 'item %r, count %r, minind %r' % (item, count, min_index)
            return count, -min_index
        # pick the highest-count/earliest item
        return max(groups, key=_auxfun)[0]
