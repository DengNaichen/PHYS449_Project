import numpy as np


def gini_impurity(y):
    m = len(y)
    # go through all classes and calculate the prob. for each class
    return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = {}

    def try_split(self, X, y, feature_index, value):
        left_mask = (X[:, feature_index] == value)
        right_mask = (X[:, feature_index] != value)
        left_impurity = gini_impurity(y[left_mask])
        right_impurity = gini_impurity(y[right_mask])
        return left_impurity, right_impurity, left_mask, right_mask

    def best_split(self, X, y):
        m, n = X.shape
        best_gain = 0
        best_feature_index = None
        best_value = None
        best_masks = None
        parent_impurity = gini_impurity(y)

        for feature_index in range(n):
            values = np.unique(X[:, feature_index])
            for value in values:
                left_impurity, right_impurity, left_mask, right_mask = self.try_split(
                    X, y, feature_index, value)
                gain = parent_impurity - (
                            left_impurity * left_mask.sum() + right_impurity * right_mask.sum()) / m
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_value = value
                    best_masks = (left_mask, right_mask)
        return best_feature_index, best_value, best_masks

    def build_tree(self, X, y, depth=0):
        num_samples_per_class, num_samples = np.unique(y, return_counts=True)
        if len(num_samples_per_class) == 1 or (self.max_depth is not None and depth == self.max_depth):
            # Return the majority class directly using counts
            majority_class_index = np.argmax(num_samples)
            return num_samples_per_class[majority_class_index]
        else:
            idx, val, masks = self.best_split(X, y)
            if idx is None:  # No split was found
                majority_class_index = np.argmax(num_samples)
                return num_samples_per_class[majority_class_index]

            left_mask, right_mask = masks
            left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
            right = self.build_tree(X[right_mask], y[right_mask], depth + 1)
            return (idx, val, left, right)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def _predict_one(self, inputs):
        node = self.tree
        while isinstance(node, tuple):
            idx, val, left, right = node
            node = left if inputs[idx] == val else right
        return node

    def predict(self, X):
        return np.array([self._predict_one(inputs) for inputs in X])


if __name__ == '__main__':
    data_number = np.array([
        [0, 0],
        [0, 1],
        [1, 2],
        [1, 0],
        [0, 2],
        [2, 1],
        [2, 0],
        [1, 1],
        [2, 2],
        [0, 1],
    ])
    # Labels
    labels = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1])

    dt = DecisionTree(max_depth=3)
    # print(dt.best_split(data_number, labels))

    dt.fit(data_number, labels)
    print(dt.predict(data_number))
    print(labels)

