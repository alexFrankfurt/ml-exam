import numpy as np
import random
from binarytree import Node


class DecisionTreeClassifier():
    def __init__(self, max_depth = 5, min_leaf = 2):
        self.root = None
        self.classes_ = []
        self.max_depth = max_depth

    # 0 <= X <= 1
    # y - classes
    def fit(self, X = [[]], y = []):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)

        self.build_tree(X, y, 0)

        return self

    def predict(self, X):
        res = []
        for x in X:
            y = self.find(x, self.root)
            res.append(y)
        return res

    def find(self, x, node):
        if node.is_leaf:
            return node.predict()
        if node.operation(x):
            return self.find(x, node.right)
        else:
            return self.find(x, node.left)

    def render(self):
        print(self.root.node)

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
            p_yes = -p_c * np.log2(p_c)
            p_total += p_yes

        return p_total

    def build_tree(self, X, y, cur_depth, node = None):
        (left_idx, rigth_idx, operation_str, operation) = self.build_node(X, y)
        if node == None:
            self.root = BinaryTreeNode()
            node = self.root
        node.set(operation, operation_str)
        print('res', (left_idx, rigth_idx, operation_str))

        if len(left_idx) == 0 or len(rigth_idx) == 0:
            print('emergency leaf')
            node.set(data = y)
            return

        (X_left, y_left, X_right, y_right) = (X[left_idx], y[left_idx], X[rigth_idx], y[rigth_idx])
        print('self-entropy', self.entropy(y_left))
        if y_left.shape[0] > 1 and self.entropy(y_left) > 0.3 and cur_depth < self.max_depth:
            node.set_left(BinaryTreeNode())
            self.build_tree(X_left, y_left, cur_depth+1, node.left)
        else:
            left_node = BinaryTreeNode(data = y_left)
            node.set_left(left_node)
            left_node.is_leaf = True

        if y_right.shape[0] > 1 and self.entropy(y_right) > 0.3 and cur_depth < self.max_depth:
            node.set_right(BinaryTreeNode())
            self.build_tree(X_right, y_right, cur_depth+1, node.right)
        else:
            right_node = BinaryTreeNode(data = y_right)
            node.set_right(right_node)
            right_node.is_leaf = True

    def build_node(self, X, y):
        y = np.array(y)
        init_entropy = self.entropy(y)
        inf_gain_list = []
        x_split_list = []
        for i in range(0,1000):
            (left_idx, right_idx, operation_str, operation_func) = self.split(X, y)
            x_split_list.append((left_idx, right_idx, operation_str, operation_func))
            (y_left, y_right) = (y[left_idx], y[right_idx])

            entropy_left = self.entropy(y_left)
            entropy_right = self.entropy(y_right)
            inf_gain = init_entropy - (entropy_left + entropy_right)
            inf_gain_list.append(inf_gain)

        best_split_arg = np.argmax(inf_gain_list)
        best_x_split = x_split_list[best_split_arg]
        print('best inf_gain:', inf_gain_list[best_split_arg])

        return best_x_split

    # function([[x1,x2,x3,y],[x1,x2,x3,y]]) => lambda
    def split(self, X, Y):
        n_features = len(X[0])
        rand_idx = random.randint(0, n_features - 1)
        limit = random.uniform(0, 1)
        signs = [">", "<", "<=", ">="]
        sign_func = [
            lambda x, y: x > y,
            lambda x, y: x < y,
            lambda x, y: x <= y,
            lambda x, y: x >= y
        ]
        sign_idx = random.randint(0, len(signs) - 1)

        operation = "x" + str(rand_idx) + signs[sign_idx] + str(round(limit, 3))
        # print('X:', X)
        # print('operation: ', operation)
        right_idx = [idx for idx, features in enumerate(X) if sign_func[sign_idx](features[rand_idx], limit)]
        left_idx = [idx for idx in range(0, len(Y)) if idx not in right_idx]
        # print('left idx: ', left_idx)
        # print('right idx: ', right_idx)

        return (left_idx, right_idx, operation, lambda x : sign_func[sign_idx](x[rand_idx], limit))


class BinaryTreeNode():
    def __init__(self, operation=None, data=None):
        self.left = None
        self.right = None
        self.operation = operation
        self._data = data
        self.is_leaf = False # Will be set to True if data is present and no operation

        if operation is not None:
            self.node = Node(operation)
        elif data is not None:
            self.node = Node("Leaf") # Placeholder for leaf node
            self.is_leaf = True
        else:
            self.node = Node("Empty") # Placeholder for a node created without initial data/operation


    def set(self, operation=None, operation_str=None, data=None):
        """
        Sets the properties of the node.
        - If operation and operation_str are provided, it's an internal node.
        - If data is provided, it's a leaf node.
        """
        if operation is not None and operation_str is not None:
            self.operation = operation
            self._data = None
            self.is_leaf = False
            self.node.value = operation_str # Use operation_str for display
        elif data is not None and data.shape[0] > 0: # Check if data is not empty
            self.operation = None
            self._data = data
            self.is_leaf = True
            # For leaf nodes, we can display the predicted class or a generic label
            (values, counts) = np.unique(data, return_counts=True)
            predicted_class = values[np.argmax(counts)]
            self.node.value = f"Leaf ({predicted_class})"
        else:
            # Default case, perhaps for nodes that are not yet fully defined or empty leaf
            self.operation = None
            self._data = None
            self.is_leaf = False
            self.node.value = "Node" # Generic label for internal or undefined nodes

    def set_right(self, binaryTreeNode):
        self.right = binaryTreeNode
        self.node.right = binaryTreeNode.node

    def set_left(self, binaryTreeNode):
        self.left = binaryTreeNode
        self.node.left = binaryTreeNode.node

    def predict(self):
        if self._data is None or self._data.shape[0] == 0:
            # Handle cases where a leaf node might not have data (e.g., empty split)
            # This might indicate an issue in build_tree logic, but for now, handle gracefully.
            return None # Or raise an error, depending on desired behavior.

        (values, counts) = np.unique(self._data, return_counts=True)
        ind = np.argmax(counts)
        return values[ind]
