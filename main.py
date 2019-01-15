from algorithms import tree
import pandas as pd
import numpy as np

from algorithms.forest import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn import preprocessing

iris = load_iris()

data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

x = data1.sample(20).values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

data_iris = df.values
x_iris = data_iris[:,0:4]
y_iris = data_iris[:,4]

print('x', x_iris)
print('y', y_iris)

tree_classifier = tree.DecisionTreeClassifier()

X = [
    [1, 0],
    [0, 0.2],
    [0.1, 0.4],
    [0.6, 0.1],
    [0.1, 0.8],
    [0.1, 0.9],
    [0.3, 0.7]
]

y = [
    0, 1, 1, 0, 2, 2, 2
]

# print('X: ', X)
# print(tree_classifier.split([[1,0],[0,2],[0.1,0.4]], [4, 3,1]))

# print(tree_classifier.build_node(X, y))
#
# print('y',tree_classifier.entropy(y))
# print('0, 1, 1, 0',tree_classifier.entropy([0, 1, 1, 0]))
# print('2, 2, 2',tree_classifier.entropy([2, 2, 2]))
# print('0,0',tree_classifier.entropy([0,0]))
# print('1,1,2,2,2',tree_classifier.entropy([1,1,2,2,2]))

tree_model = tree_classifier.fit(x_iris, y_iris)
tree_classifier.render()

res = tree_model.predict(x_iris)
print('res:', res)


RandomForestRegressor(X, y, 20, 3, 12)
