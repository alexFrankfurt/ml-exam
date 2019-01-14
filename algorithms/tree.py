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
