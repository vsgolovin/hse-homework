import copy
from typing import NoReturn, List
import numpy as np


class Module:
    """
    Абстрактный класс. Его менять не нужно.
    """
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, d):
        raise NotImplementedError()
        
    def update(self, alpha):
        pass


class Linear(Module):
    """
    Линейный полносвязный слой.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int 
            Размер выхода.
    
        Notes
        -----
        W и b инициализируются случайно.
        """
        # Xavier initialization
        std = np.sqrt(in_features)
        self.weights = np.random.randn(in_features, out_features) / std
        self.biases = np.random.randn(out_features) / std
        self.inputs = None
        self.grad = None    # output gradients

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).

        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        self.inputs = copy.deepcopy(x)
        return x @ self.weights + self.biases

    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        self.grad = copy.deepcopy(d)
        return d @ self.weights.T

    def update(self, alpha: float) -> NoReturn:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """
        self.weights -= alpha * (self.inputs.T @ self.grad) / len(self.inputs)
        self.biases -= alpha * self.grad.mean(axis=0)


class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU.
    """
    def __init__(self):
        self.mask = None  # positive inputs

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
    
        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        self.mask = x > 0.
        y = np.zeros_like(x)
        y[self.mask] = x[self.mask]
        return y

    def backward(self, d) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        d_new = np.zeros_like(d)
        d_new[self.mask] = d[self.mask]
        return d_new


class Softmax(Module):
    """
    Слой, соответствующий функции активации Softmax.
    """
    def __init__(self):
        self.jac = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Softmax(x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.

        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        exp_x = np.exp(x - np.max(x))
        s = exp_x / exp_x.sum(axis=1).reshape((-1, 1))
        self.jac = s * (1 - s)
        return s

    def backward(self, d) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        return d * self.jac


class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40, alpha: float = 0.01):
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и 
            описывающий слои нейронной сети. 
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обученияю
        alpha : float
            Cкорость обучения.
        """
        self.modules = modules
        self.modules.append(Softmax())
        self.epochs = epochs
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size=32) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох. 
        В каждой эпохе необходимо использовать cross-entropy loss для обучения, 
        а так же производить обновления не по одному элементу, а используя батчи.

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        batch_size : int
            Размер батча.
        """
        num_samples = len(X)
        assert num_samples >= batch_size
        num_steps = int(np.ceil(num_samples / batch_size))
        y = np.asarray(y)

        for i in range(self.epochs):
            for _ in range(num_steps):
                # select batch
                inds = np.random.choice(np.arange(num_samples), batch_size, replace=False)
                x_i = X[inds]
                y_i = y[inds]

                # forward pass
                for module in self.modules:
                    x_i = module.forward(x_i)

                # cross-entropy gradient
                ix = np.arange(batch_size), y_i
                grad = np.zeros_like(x_i)  
                grad[ix] = -1 / x_i[ix]

                # backward pass
                for module in reversed(self.modules):
                    grad = module.backward(grad)
                    module.update(self.alpha)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.
        
        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)
        
        """
        for module in self.modules:
            X = module.forward(X)
        return X

    def predict(self, X) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.
        
        Return
        ------
        np.ndarray
            Вектор предсказанных классов
        
        """
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)


if __name__ == '__main__':
    p = MLPClassifier([
        Linear(4, 64),
        ReLU(),
        Linear(64, 64),
        ReLU(),
        Linear(64, 2)
    ], epochs=100)

    X = np.random.randn(50, 4)
    y = [(0 if x[0] > x[2]**2 or x[3]**3 > 0.5 else 1) for x in X]
    p.fit(X, y)
    print((p.predict(X) == y).sum() / len(y))
