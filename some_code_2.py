import numpy as np


class MyLinearRegression:
    def __init__(self):
        self._w = None
        self._X_train = None
        self._y_train = None

    def __str__(self):
        return f"MyLinearRegression"

    def _get_loss(self, X: np.array, y: np.array):
        y_pred = np.dot(X, self._w)
        return -2 * np.dot(X.T, (y - y_pred)) / X.size

    def _get_points_by_batch(self, batch: int):
        indexes = np.random.randint(low=0, high=len(self._X_train) - 1, size=batch)
        X_i = self._X_train[indexes]
        y_i = self._y_train[indexes]
        return X_i, y_i

    def _stochastic_gd(self, batch: int, lr: float, epoch: int):
        for _ in range(epoch):
            X_i, y_i = self._get_points_by_batch(batch)
            step = self._get_loss(X_i, y_i)
            self._w = self._w - lr * step

    def fit(
        self,
        X_train,
        y_train,
        w: np.array = None,
        batch: int = 100,
        lr: float = 0.9,
        epoch: int = 1000,
    ):
        self._X_train, self._y_train = np.array(X_train), np.array(y_train)
        if w is None:
            self._w = np.zeros(X_train.shape[1])
        if batch is None:
            batch = int(X_train.size * 0.3)

        self._stochastic_gd(batch, lr, epoch)

    def predict(self, X: np.array):
        return np.dot(X, self._w)
