from algorithms import tree

tree_classifier = tree.DecisionTreeClassifier()

# print(tree_classifier.entropy([1,1,0,0,2,2,3,3]))
# print(tree_classifier.entropy([1,0,2,2,3,3]))
# print(tree_classifier.entropy([1,0]))

print('data: ', [[1,0],[0,2],[0.1,0.4]])
print(tree_classifier.split([[1,0],[0,2],[0.1,0.4]], [4, 3,1]))
