import numpy as np

class DecisionTreeClassifier():
    def __init__(self, max_height = 5):
        self.classes_ = []
        self.max_height = max_height

    # 0 <= X <= 1
    # y - classes
    def fit(self, X = [[]], y = []):
        self.classes_ = np.unique(y)

        self.build_tree(X, y, 0)

        print('fit')

    def predict(self, X):
        print('predict')

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

    def build_tree(self, X, y, cur_height):
        (X_left, y_left, X_right, y_right) = self.build_node(X, y)
        if y_left.shape > 3 and self.entropy(y_left) > 0.3 and cur_height < self.max_height:
            self.build_tree(X_left, y_left, cur_height+1)

        if y_right.shape > 3 and self.entropy(y_right) > 0.3 and cur_height < self.max_height:
            self.build_tree(X_right, y_right, cur_height+1)

    def build_node(self, X, y):
        init_entropy = self.entropy(y)
        inf_gain_list = []
        x_split_func_list = []
        for i in range(0,100):
            x_split_func = selt._split(X, y)
            x_split_func_list.push(x_split_func)
            (y1, y2, _) = x_split_func()
            entropy_left = self.entropy(y1)
            entropy_right = self.entropy(y2)
            inf_gain = init_entropy - (entropy_left + entropy_right)
            inf_gain_list.push(inf_gain)

        best_split_arg = np.argmax(inf_gain_list)
        best_x_split = x_split_func_list[best_split_arg]
        (_, _, node) = best_x_split()

        return ()

    # function([[x1,x2,x3,y],[x1,x2,x3,y]]) => lambda
    def _split(self, X, y):
        x_split = x3 < rand(0,1)
        # lambda => [y1, y2, string = x1 < 0.7]
        return function(x_index = 1, n_rand = 0.7) {

        }
