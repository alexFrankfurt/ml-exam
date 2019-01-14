from algorithms import tree

tree_classifier = tree.DecisionTreeClassifier()

print(tree_classifier.binary_entropy([1,1,1,1,1,1,1,1,1,0,0,0,0,0]))
print(tree_classifier.entropy([1,1,1,0,0,0,2,2,2]))
