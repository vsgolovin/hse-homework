from typing import Callable, Union, Any
import numpy as np


def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    _, counts = np.unique(x, return_counts=True)
    p = counts / len(x)
    return np.sum(p * (1 - p))


def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    _, counts = np.unique(x, return_counts=True)
    p = counts / len(x)
    return -np.sum(p * np.log2(p))


def gain(left_y: np.ndarray, right_y: np.ndarray,
         criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    y = np.concatenate([left_y, right_y])
    return criterion(y) - (len(left_y) * criterion(left_y)
                           + len(right_y) * criterion(right_y)) / len(y)


def get_criterion(name: str) -> Callable:
    name = name.lower()
    if name == 'gini':
        return gini
    assert name == 'entropy'
    return entropy


class Node:
    def __init__(self, criterion: Union[Callable, str]):
        if isinstance(criterion, str):
            criterion = get_criterion(criterion)
        self.criterion = criterion
        self.split_dim = None
        self.label = None
        self.left = None
        self.right = None

    def _assign_class(self, y):
        labels, counts = np.unique(y, return_counts=True)
        i = np.argmax(counts)
        self.label = labels[i]

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    def split(self, X: np.ndarray, y: np.ndarray, levels_left: int,
              min_samples_leaf: int, max_features: int):
        # check if reached max depth
        if levels_left == 0:
            self._assign_class(y)
            return

        # find best split
        dim_best = None
        gain_best = -np.inf
        for dim in np.random.choice(np.arange(X.shape[1]), size=max_features):
            ix = X[:, dim] == 0  # binary features
            samples_left = np.sum(ix)
            if min(samples_left, len(X) - samples_left) < min_samples_leaf:
                continue  # not enough samples in leaf
            g = gain(y[ix], y[~ix], self.criterion)
            if g > gain_best:
                dim_best = dim
                gain_best = g

        # check if managed to split
        if dim_best is None:
            self._assign_class(y)
            return

        # create leaves
        self.split_dim = dim_best
        ix = X[:, dim_best] == 0
        self.left = Node(self.criterion)
        self.left.split(X[ix, :], y[ix], levels_left - 1,
                        min_samples_leaf, max_features)
        self.right = Node(self.criterion)
        self.right.split(X[~ix, :], y[~ix], levels_left - 1,
                         min_samples_leaf, max_features)

    def _predict(self, x: np.ndarray) -> Any:
        if self.is_leaf:
            return self.label
        if x[self.split_dim] == 0:
            return self.left._predict(x)
        return self.right._predict(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        assert X.ndim == 2

        # traverse tree for every sample
        for x in X:
            predictions.append(self._predict(x))
        return np.array(predictions)


class DecisionTree:
    def __init__(self, X, y, criterion="gini", max_depth=None,
                 min_samples_leaf=1, max_features="auto"):
        # choose samples with bagging
        self.inds_samples = np.random.choice(
            np.arange(len(X)), size=len(X), replace=True)
        X = X[self.inds_samples, :]
        y = y[self.inds_samples]

        # choose features
        if max_features == "auto":
            max_features = int(round(np.sqrt(X.shape[1])))
        assert isinstance(max_features, int) and 0 < max_features <= X.shape[1]

        # create tree and perform splits recursively
        self.root = Node(criterion)
        if max_depth is None:
            max_depth = -1
        self.root.split(X, y, max_depth, min_samples_leaf, max_features)

    def predict(self, X, soft=False):
        """
        Returns binary class labels (if `soft=False`) or probabilities for
        every sample in 2D array `X`.
        """
        return self.root.predict(X)


class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1,
                 max_features="auto", n_estimators=10):
        self.estimators = [None] * n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def fit(self, X, y):
        for i in range(len(self.estimators)):
            self.estimators[i] = DecisionTree(
                X=X,
                y=y,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )

    def predict(self, X):
        predictions = []
        for estimator in self.estimators:
            predictions.append(estimator.predict(X))
        most_frequent = [None] * len(X)
        for i in range(len(X)):
            sample_predictions = np.array([p[i] for p in predictions])
            labels, counts = np.unique(sample_predictions, return_counts=True)
            most_frequent[i] = labels[np.argmax(counts)]
        return np.array(most_frequent)


def synthetic_dataset(size):
    X = [(np.random.randint(0, 2), np.random.randint(0, 2), i % 6 == 3, 
          i % 6 == 0, i % 3 == 2, np.random.randint(0, 2)) for i in range(size)]
    y = [i % 3 for i in range(size)]
    return np.array(X), np.array(y)


if __name__ == '__main__':
    X, y = synthetic_dataset(1000)
    rfc = RandomForestClassifier(n_estimators=100, max_features='auto')
    rfc.fit(X, y)
    print("Accuracy:", np.mean(rfc.predict(X) == y))

    # tree = DecisionTree(X, y, min_samples_leaf=30, max_depth=6)
    # print((tree.predict(X) == y).sum() / len(y))
