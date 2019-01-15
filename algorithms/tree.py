import numpy as np
from random import choice, randint, uniform
from anytree import Node, RenderTree


class DecisionTreeClassifier():
    def __init__(self, max_depth = 5, min_leaf = 2):
        self.root_render = None
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
        # print('tree:', self.root_render)

    def predict(self, X):
        res = []
        for x in X:
            y = self.find(x, self.root)
            res.append(y)
        return res

    def find(self, x, node):
        if isinstance(node, BinaryTreeLeaf):
            return node.predict()
        if node.operation(x):
            return self.find(x, node.right)
        else:
            return self.find(x, node.left)

    def render(self):
        for pre, fill, node in RenderTree(self.root_render):
            print("%s%s" % (pre, node.name))

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

    def build_tree(self, X, y, cur_depth, parent_render = None, parent = None):
        (left_idx, rigth_idx, cur_node_render, operation) = self.build_node(X, y, parent_render)
        if self.root_render == None:
            self.root_render = cur_node_render
            self.root = BinaryTreeNode()
            parent = self.root
        parent.set_operation(operation)

        (X_left, y_left, X_right, y_right) = (X[left_idx], y[left_idx], X[rigth_idx], y[rigth_idx])
        if y_left.shape[0] > 1 and self.entropy(y_left) > 0.3 and cur_depth < self.max_depth:
            parent.left = BinaryTreeNode()
            self.build_tree(X_left, y_left, cur_depth+1, cur_node_render, parent.left)
        else:
            Node(y[left_idx], cur_node_render)
            parent.left = BinaryTreeLeaf(y_left)

        if y_right.shape[0] > 1 and self.entropy(y_right) > 0.3 and cur_depth < self.max_depth:
            parent.right = BinaryTreeNode()
            self.build_tree(X_right, y_right, cur_depth+1, cur_node_render, parent.right)
        else:
            Node(y[rigth_idx], cur_node_render)
            parent.right = BinaryTreeLeaf(y_right)

    def build_node(self, X, y, parent_render):
        y = np.array(y)
        init_entropy = self.entropy(y)
        inf_gain_list = []
        x_split_list = []
        for i in range(0,1000):
            (left_idx, right_idx, operation_str, operation_func) = self.split(X, y)
            x_split_list.append((left_idx, right_idx, operation_str))
            (y_left, y_right) = (y[left_idx], y[right_idx])

            entropy_left = self.entropy(y_left)
            entropy_right = self.entropy(y_right)
            inf_gain = init_entropy - (entropy_left + entropy_right)
            inf_gain_list.append(inf_gain)

        best_split_arg = np.argmax(inf_gain_list)
        best_x_split = x_split_list[best_split_arg]
        operation_node = Node(best_x_split[2], parent_render)

        return (left_idx, right_idx, operation_node, operation_func)

    # function([[x1,x2,x3,y],[x1,x2,x3,y]]) => lambda
    def split(self, X, Y):
        n_features = len(X[0])
        rand_idx = randint(0, n_features - 1)
        limit = uniform(0, 1)
        signs = [">", "<", "<=", ">="]
        sign_func = [
            lambda x, y: x > y,
            lambda x, y: x < y,
            lambda x, y: x <= y,
            lambda x, y: x >= y
        ]
        sign_idx = randint(0, len(signs) - 1)

        operation = "x" + str(rand_idx) + signs[sign_idx] + str(limit)
        # print('X:', X)
        # print('operation: ', operation)
        right_idx = [idx for idx, features in enumerate(X) if sign_func[sign_idx](features[rand_idx], limit)]
        left_idx = [idx for idx in range(0, len(Y)) if idx not in right_idx]
        # print('left idx: ', left_idx)
        # print('right idx: ', right_idx)

        return (left_idx, right_idx, operation, lambda x : sign_func[sign_idx](x[rand_idx], limit))


class BinaryTreeNode():
    def __init__(self, operation = None):
        self.left = None
        self.right = None
        self.operation = operation

    def set_operation(self, operation):
        self.operation = operation


class BinaryTreeLeaf():
    def __init__(self, data):
        self.data = np.array(data)

    def predict(self):
        (values, counts) = np.unique(self.data, return_counts=True)
        ind = np.argmax(counts)
        return values[ind]
