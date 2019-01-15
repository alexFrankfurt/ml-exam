from algorithms import tree, forest
import pandas as pd
import numpy as np

from algorithms.forest import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn import preprocessing

iris = load_iris()

data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

x = data1.sample(100).values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

data_iris = df.values
x_iris = data_iris[0:50,0:4]
y_iris = data_iris[0:50,4]

x_test = data_iris[50:100,0:4]
y_test = data_iris[50:100,4]

print('x', x_iris)
print('y', y_iris.tolist())

tree_classifier = tree.DecisionTreeClassifier()

tree_model = tree_classifier.fit(x_iris, y_iris)
tree_classifier.render()

y_ = tree_model.predict(x_test)
print('y_:', y_)

res = y_test == y_
# print('res:', res)
acc = np.sum(res) / len(y_)
print('accuracy:', acc)


forest_classifier = forest.RandomForestRegressor(x_iris, y_iris, n_trees=10, n_features=3, sample_sz=10)
forest_res = forest_classifier.best_guess
print(forest_res)
