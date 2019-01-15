import numpy as np
from random import choice, randint, uniform
from anytree import Node


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

        # print('tree:', self.root)

    def predict(self, X):
        print('predict')

    def get_root(self):
        return self.root

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

    def build_tree(self, X, y, cur_depth, parent = None):
        (left_idx, rigth_idx, cur_node) = self.build_node(X, y, parent or self.root)
        if self.root == None:
            self.root = cur_node
        (X_left, y_left, X_right, y_right) = (X[left_idx], y[left_idx], X[rigth_idx], y[rigth_idx])
        if y_left.shape[0] > 1 and self.entropy(y_left) > 0.3 and cur_depth < self.max_depth:
            self.build_tree(X_left, y_left, cur_depth+1, cur_node)
        else:
            Node(y[left_idx], cur_node)

        if y_right.shape[0] > 1 and self.entropy(y_right) > 0.3 and cur_depth < self.max_depth:
            self.build_tree(X_right, y_right, cur_depth+1, cur_node)
        else:
            Node(y[rigth_idx], cur_node)

    def build_node(self, X, y, parent):
        y = np.array(y)
        init_entropy = self.entropy(y)
        inf_gain_list = []
        x_split_list = []
        for i in range(0,1000):
            (left_idx, right_idx, operation) = self.split(X, y)
            x_split_list.append((left_idx, right_idx, operation))
            (y_left, y_right) = (y[left_idx], y[right_idx])

            entropy_left = self.entropy(y_left)
            entropy_right = self.entropy(y_right)
            inf_gain = init_entropy - (entropy_left + entropy_right)
            inf_gain_list.append(inf_gain)

        best_split_arg = np.argmax(inf_gain_list)
        best_x_split = x_split_list[best_split_arg]
        operation_node = Node(best_x_split[2], parent)

        return (left_idx, right_idx, operation_node)

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
        left_idx = [idx for idx, features in enumerate(X) if sign_func[sign_idx](features[rand_idx], limit)]
        right_idx = [idx for idx in range(0, len(Y)) if idx not in left_idx]
        # print('left idx: ', left_idx)
        # print('right idx: ', right_idx)

        return (left_idx, right_idx, operation)
