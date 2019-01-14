import numpy as np

class DecisionTreeClassifier():
    def __init__(self):
        self.classes_ = []
        print('__init__')

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        print('fit')

    def predict(self, X):
        print('predict')

    def binary_entropy(self, y_binary):
        y_binary = np.array(y_binary)
        p_c = np.sum(y_binary) / y_binary.shape[0]
        p_yes = -p_c * np.log2(p_c)
        p_no = -(1 - p_c) * np.log2(1 - p_c)
        return p_yes + p_no

    def entropy(self, classes):
        classes = np.array(classes)
        uniq_classes = np.unique(classes)
        p_total = 0

        for c in uniq_classes:
            condition = classes == c
            c_count = np.count_nonzero(condition)
            p_c = c_count / classes.shape[0]
            print('p_c', p_c)
            p_yes = -p_c * np.log2(p_c)
            print('p_yes', p_yes)
            p_total += p_yes

        return p_total
