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
print('y', y_iris.tolist())

tree_classifier = tree.DecisionTreeClassifier()

tree_model = tree_classifier.fit(x_iris, y_iris)
tree_classifier.render()

res = tree_model.predict(x_iris)
print('res:', res)


RandomForestRegressor(X, y, 20, 3, 12)
