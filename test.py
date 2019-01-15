from algorithms import tree

tree_classifier = tree.DecisionTreeClassifier()

# print(tree_classifier.entropy([1,1,0,0,2,2,3,3]))
# print(tree_classifier.entropy([1,0,2,2,3,3]))
# print(tree_classifier.entropy([1,0]))

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

print(tree_classifier.entropy([0.5, 0.5, 1, 1, 1.,  1, 0.5, 0.5]))
