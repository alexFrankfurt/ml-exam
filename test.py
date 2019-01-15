from algorithms import tree

tree_classifier = tree.DecisionTreeClassifier()

# print(tree_classifier.entropy([1,1,0,0,2,2,3,3]))
# print(tree_classifier.entropy([1,0,2,2,3,3]))
# print(tree_classifier.entropy([1,0]))

print(tree_classifier.entropy([]))
